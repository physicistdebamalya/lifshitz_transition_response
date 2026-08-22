// main cpp file for tight binding Altermagnetic model

#include <fstream>
#include "necessary_functions.h"
#include "unsupported/Eigen/KroneckerProduct"

// using namespace std;
// using namespace Eigen;

// *************************************************************************

const std::string System = "Response of Lifshitz transition in a two-band model";

// Every column of task.csv this binary reads. A header cell absent from this
// list is a hard error rather than an ignored column -- see
// require_known_columns(). analysis/models.py:verify_cpp_task_columns()
// parses this list out of the source and asserts it equals
// ModelSchema.task_columns, which is the only thing standing between the two
// sides silently drifting (C++ cannot import the schema).
const std::vector<std::string> KNOWN_COLUMNS = {
    "System", "alpha", "beta", "delta", "m", "Temperature", "ef", "dephase",
    "N1", "N2", "b1_0", "b1_1", "b2_0", "b2_1", "k_origin_0", "k_origin_1",
    "TIME_DIVISION", "ETP",
    "A0X", "nuX", "sigmaX", "ntt_sigmaX", "delayX", "durationX",
    "number_of_cycleX", "pulse_typeX", "pulse_wrapperX",
    "A0Y", "nuY", "sigmaY", "ntt_sigmaY", "delayY", "durationY",
    "number_of_cycleY", "pulse_typeY", "pulse_wrapperY"};

// One parameter point. POD, so it can be MPI_Bcast as raw bytes.
//
// Deliberately NO member initialisers. task.csv is always written by exactly
// one function (workflow/task_schema.py:write_task_csv) and always carries
// every column, so a field left unset could only ever be reached by a bug --
// and a default is the worst possible answer to a bug, because it produces
// entirely plausible numbers for a point nobody asked for.
struct TaskSpec
{
    double alpha, beta, delta, m;
    double Temperature, ef, dephase;
    int N1, N2;
    double b1[2], b2[2], k_origin[2]; // raw arrays keep this trivially copyable
    int TIME_DIVISION;
    double ETP;
    double A0X, nuX, sigmaX, ntt_sigmaX, delayX, durationX;
    int number_of_cycleX;
    PulseKind pulse_typeX, pulse_wrapperX;
    double A0Y, nuY, sigmaY, ntt_sigmaY, delayY, durationY;
    int number_of_cycleY;
    PulseKind pulse_typeY, pulse_wrapperY;
};

static_assert(std::is_trivially_copyable_v<TaskSpec>,
              "TaskSpec is MPI_Bcast as raw bytes; it must stay POD");

TaskSpec load_task(const std::string &path)
{
    const TaskRow row = read_single_task_row(path, KNOWN_COLUMNS);

    // The binary and the task file are two independent files in a run
    // directory now -- where a rewritten-and-recompiled source used to make a
    // two_band point runnable only by a two_band binary. Copying the wrong
    // one, or pointing this at another directory's task.csv, would otherwise
    // produce a complete and plausible data.csv labelled with the wrong
    // System. This is the check that closes that.
    if (row.text("System") != System)
        throw std::runtime_error(
            path + ": the System column says \"" + row.text("System") +
            "\" but this binary is \"" + System + "\".");

    TaskSpec T;
    T.alpha = row.dbl("alpha");
    T.beta = row.dbl("beta");
    T.delta = row.dbl("delta");
    T.m = row.dbl("m");
    T.Temperature = row.dbl("Temperature");
    T.ef = row.dbl("ef");
    T.dephase = row.dbl("dephase");
    T.N1 = row.integer("N1");
    T.N2 = row.integer("N2");
    T.b1[0] = row.dbl("b1_0");
    T.b1[1] = row.dbl("b1_1");
    T.b2[0] = row.dbl("b2_0");
    T.b2[1] = row.dbl("b2_1");
    T.k_origin[0] = row.dbl("k_origin_0");
    T.k_origin[1] = row.dbl("k_origin_1");
    T.TIME_DIVISION = row.integer("TIME_DIVISION");
    T.ETP = row.dbl("ETP");

    T.A0X = row.dbl("A0X");
    T.nuX = row.dbl("nuX");
    T.sigmaX = row.dbl("sigmaX");
    T.ntt_sigmaX = row.dbl("ntt_sigmaX");
    T.delayX = row.dbl("delayX");
    T.durationX = row.dbl("durationX");
    T.number_of_cycleX = row.integer("number_of_cycleX");
    T.pulse_typeX = row.kind("pulse_typeX");
    T.pulse_wrapperX = row.kind("pulse_wrapperX");

    T.A0Y = row.dbl("A0Y");
    T.nuY = row.dbl("nuY");
    T.sigmaY = row.dbl("sigmaY");
    T.ntt_sigmaY = row.dbl("ntt_sigmaY");
    T.delayY = row.dbl("delayY");
    T.durationY = row.dbl("durationY");
    T.number_of_cycleY = row.integer("number_of_cycleY");
    T.pulse_typeY = row.kind("pulse_typeY");
    T.pulse_wrapperY = row.kind("pulse_wrapperY");

    // Guards that a constexpr made unreachable by inspection and a runtime
    // value does not.
    if (T.N1 < 2 || T.N2 < 2)
        throw std::runtime_error(path + ": N1 and N2 must be >= 2 (the k-grid "
                                        "divides by N-1)");
    if (T.TIME_DIVISION < 1)
        throw std::runtime_error(path + ": TIME_DIVISION must be >= 1");

    return T;
}

// ----------------------------------------------------------------------------------

// Basis indices for the 2-band Hamiltonian. Positional and generic -- this
// model has no spin or particle-hole structure, so a label like Up/Dn
// borrowed from a BdG model would assert a symmetry that isn't here.
enum Basis
{
    State0,
    State1
};

// Peierls-shifted momentum. Computed once, in one place, and reused by both
// Hamiltonian and velocity_operator below.
inline Eigen::Vector2d shifted_momentum(const Eigen::Vector2d &k, const Eigen::Vector2d &A)
{
    return k - constants::e * A;
}

// The band coefficients, previously namespace-scope constexpr.
//
// EVERY EXPRESSION BELOW IS CHARACTER-IDENTICAL TO THE CONSTEXPR VERSION.
// Only the storage class changed. That is deliberate: a hand-hoisted
// compound constant (2.0*alpha, -z*beta) would change the association of the
// floating-point arithmetic, and `(-z)*beta*k1` is not bit-identical to
// `-(z*beta*k1)` when the real part is a signed zero. Leaving the
// expressions alone makes "the numbers cannot have changed" a matter of
// reading the diff rather than an argument about IEEE-754. GCC hoists
// loop-invariant subexpressions itself; if a benchmark ever says otherwise,
// measure first and hoist second.
struct BandCoeffs
{
    double alpha, beta, delta, m;

    // The Hamiltonian has exactly one off-diagonal entry (State0,State1); the
    // other one is its conjugate. off_diag and D_off_diag are the entry and
    // its k-gradient -- update both together, since velocity_operator is
    // built entirely from D_off_diag rather than re-derived independently.
    inline c_d off_diag(const Eigen::Vector2d &k) const
    {
        return alpha * (k[0] * k[0]) - delta - constants::z * beta * k[1];
    }

    inline Eigen::Vector2cd D_off_diag(const Eigen::Vector2d &k) const
    {
        return Eigen::Vector2cd{2.0 * alpha * k[0], -constants::z * beta};
    }
};

// Functors rather than free functions, because the parameters they close over
// are no longer compile-time constants. necessary_functions is templated on
// HFunc/voFunc and only ever calls `Ham(k, A)`, so a class with a const
// operator() substitutes with ZERO changes to the engine.
struct HamiltonianOp
{
    BandCoeffs c;

    Eigen::Matrix2cd operator()(const Eigen::Vector2d &k, const Eigen::Vector2d &A) const
    {
        Eigen::Matrix2cd Ham = Eigen::Matrix2cd::Zero();
        const Eigen::Vector2d p = shifted_momentum(k, A);

        Ham(State0, State0) = c.m;
        Ham(State1, State1) = -c.m;

        const c_d h01 = c.off_diag(p);
        Ham(State0, State1) = h01;
        Ham(State1, State0) = std::conj(h01);

        return Ham;
    }
};

struct VelocityOp
{
    BandCoeffs c;

    std::array<Eigen::Matrix2cd, 2> operator()(const Eigen::Vector2d &k, const Eigen::Vector2d &A) const
    {
        std::array<Eigen::Matrix2cd, 2> co = {Eigen::Matrix2cd::Zero(), Eigen::Matrix2cd::Zero()};
        const Eigen::Vector2cd D = c.D_off_diag(shifted_momentum(k, A));

        co[0](State0, State1) = D[0];
        co[0](State1, State0) = std::conj(D[0]);

        co[1](State0, State1) = D[1];
        co[1](State1, State0) = std::conj(D[1]);

        return co;
    }
};

// *****************************************************************************

// The Lindblad operator SHAPES. Structure, not parameters -- only their
// `dephase` prefactor is a parameter, so Ls itself is built in main() once
// dephase is known.
const Eigen::Matrix2cd L1{{c_d(1.0), c_d(0.0)},
                          {c_d(0.0), c_d(0.0)}};

const Eigen::Matrix2cd L2{{c_d(0.0), c_d(0.0)},
                          {c_d(0.0), c_d(1.0)}};

int main(int argc, char *argv[])
{
    const int nthreads = max_threads();

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // 2. Safety check: Did the cluster actually give us thread support?
    if (provided < MPI_THREAD_FUNNELED)
    {
        if (world_rank == 0)
            std::cerr << "Error: MPI provides insufficient thread support\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ---- the one parameter point ------------------------------------------
    // Rank 0 reads and validates; everyone else receives the parsed struct.
    // Not "every rank opens the file": N ranks reading a shared Lustre file
    // can disagree (a partial write, a stale attribute cache) and nothing
    // downstream would notice -- the run would integrate different
    // Hamiltonians on different ranks and MPI_Reduce them together into a
    // plausible answer. One reader, one interpretation, by construction.
    const std::string task_file = (argc > 1) ? argv[1] : "task.csv";
    TaskSpec T{};
    if (world_rank == 0)
    {
        try
        {
            T = load_task(task_file);
        }
        catch (const std::exception &e)
        {
            std::cerr << "FATAL: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(&T, sizeof(TaskSpec), MPI_BYTE, 0, MPI_COMM_WORLD);

    // ---- unpack ------------------------------------------------------------
    // Named exactly as the constexpr globals they replace, so every line that
    // follows -- print_core_parameters, the PulseType constructors, the
    // KBlock loop, the A0X != 0.0 tests, the TIME_DIVISION arithmetic -- is
    // UNCHANGED TEXT. In particular print_core_parameters below is not edited
    // by one character, which is what keeps parameters.txt byte-identical and
    // keeps analysis/models.py:cpp_emitted_keys()'s source slice valid.
    const double alpha = T.alpha, beta = T.beta, delta = T.delta, m = T.m;
    const double Temperature = T.Temperature, ef = T.ef, dephase = T.dephase;
    const double ETP = T.ETP;
    const int TIME_DIVISION = T.TIME_DIVISION;
    const int N1 = T.N1, N2 = T.N2;
    const Eigen::Vector2d b1{T.b1[0], T.b1[1]};
    const Eigen::Vector2d b2{T.b2[0], T.b2[1]};
    const Eigen::Vector2d k_origin{T.k_origin[0], T.k_origin[1]};

    const double A0X = T.A0X, nuX = T.nuX, sigmaX = T.sigmaX;
    const double ntt_sigmaX = T.ntt_sigmaX, delayX = T.delayX, durationX = T.durationX;
    const int number_of_cycleX = T.number_of_cycleX;
    const PulseKind pulse_typeX = T.pulse_typeX, pulse_wrapperX = T.pulse_wrapperX;

    const double A0Y = T.A0Y, nuY = T.nuY, sigmaY = T.sigmaY;
    const double ntt_sigmaY = T.ntt_sigmaY, delayY = T.delayY, durationY = T.durationY;
    const int number_of_cycleY = T.number_of_cycleY;
    const PulseKind pulse_typeY = T.pulse_typeY, pulse_wrapperY = T.pulse_wrapperY;

    const std::vector<Eigen::Matrix2cd, Eigen::aligned_allocator<Eigen::Matrix2cd>>
        Ls{dephase * L1, dephase * L2};

    auto print_core_parameters = [&](std::ostream &os)
    {
        os << System << "\n";
        os << "alpha = " << alpha << "\n";
        os << "beta = " << beta << "\n";
        os << "delta = " << delta << "\n";
        os << "m = " << m << "\n";
        os << "Temperature = " << Temperature << "\n";
        os << "ef = " << ef << "\n";
        os << "dephase = " << dephase << "\n";

        os << "N1 = " << N1 << "\n";
        os << "N2 = " << N2 << "\n";

        os << "b1[0] = " << b1[0] << ", b1[1] = " << b1[1] << "\n";
        os << "b2[0] = " << b2[0] << ", b2[1] = " << b2[1] << "\n";
        os << "k_origin[0] = " << k_origin[0] << ", k_origin[1] = " << k_origin[1] << "\n";

        os << "TIME_DIVISION = " << TIME_DIVISION << "\n";
        os << "ETP = " << ETP << "\n";
    };

    // Built once, early, before any output -- so both the startup echo below
    // and the parameters.txt write later can call the same
    // PulseType::printPulseParams() on the same objects, matching AM3.cpp's
    // ordering. This has its own narrow try/catch rather than sharing the
    // big one further down: PulseType's constructor throws
    // std::invalid_argument on an unsupported pulse_type/pulse_wrapper
    // combination, and a run that fails here never actually started, so it
    // should not print "STARTING SIMULATION" first.
    PulseType PulseObjectX, PulseObjectY;
    try
    {
        PulseObjectX = PulseType(pulse_typeX, pulse_wrapperX, A0X, sigmaX, ntt_sigmaX, nuX, ETP, delayX, number_of_cycleX, durationX);
        PulseObjectY = PulseType(pulse_typeY, pulse_wrapperY, A0Y, sigmaY, ntt_sigmaY, nuY, ETP, delayY, number_of_cycleY, durationY);
    }
    catch (const std::exception &e)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cerr << "[Rank " << rank << "] FATAL EXCEPTION: \n"
                  << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (world_rank == 0)
    {
        if (System != "Response of Lifshitz transition in a two-band model")
        {
            std::cerr << "System mismatch.\n It should be: ** Response of Lifshitz transition in a two-band model **\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Using %d threads.\n", nthreads);
        std::cout << "\n--- STARTING SIMULATION ---\n";

        std::cout << std::setprecision(6);   // match parameters.txt exactly
        print_core_parameters(std::cout);
        PulseObjectX.printPulseParams(std::cout, "X");
        PulseObjectY.printPulseParams(std::cout, "Y");
        std::cout << "\n-----------------------------------------------------------------------------\n";
        printCurrentDateTime();
        std::cout << "\n-----------------------------------------------------------------------------\n";
        std::cout << std::flush;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    std::unique_ptr<ProgramTimer> timer;

    if (world_rank == 0)
    {
        timer = std::make_unique<ProgramTimer>();
    }

    try
    {
        // PulseObjectX / PulseObjectY were already built above.

        double initial_time = 0.0;
        double final_time = 0.0;

        if (A0X != 0.0 && A0Y != 0.0)
        {
            initial_time = std::min(PulseObjectX.ti, PulseObjectY.ti);
            final_time = std::max(PulseObjectX.tf, PulseObjectY.tf);
        }
        else if (A0X != 0.0)
        {
            initial_time = PulseObjectX.ti;
            final_time = PulseObjectX.tf;
        }
        else if (A0Y != 0.0)
        {
            initial_time = PulseObjectY.ti;
            final_time = PulseObjectY.tf;
        }
        else
        {
            if (world_rank == 0)
                std::cerr << "Error: At least one of the pulses should be non-zero\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        const int calculated_divisions = static_cast<int>(std::floor(TIME_DIVISION * (final_time - initial_time)));
        const int division = std::max(calculated_divisions, TIME_DIVISION);

        const double dt = (std::abs(final_time - initial_time)) / static_cast<double>(division);

        std::vector<double> Time(division + 1);

        for (int i = 0; i <= division; i++)
        {
            Time[i] = initial_time + (i * dt);
        }

        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> KBlock;
        KBlock.reserve(N1 * N2);
        for (int n1 = 0; n1 < N1; ++n1)
        {
            const double d1 = static_cast<double>(n1) / static_cast<double>(N1 - 1);
            for (int n2 = 0; n2 < N2; ++n2)
            {
                const double d2 = static_cast<double>(n2) / static_cast<double>(N2 - 1);
                KBlock.emplace_back(k_origin + (d1 * b1) + (d2 * b2));
            }
        }

        D2Pulse pulse(PulseObjectX, PulseObjectY);
        // if (!(pulse.A(Time[0])).isApprox(Eigen::Vector2d::Zero(), 1e-12))
        if (pulse.A(Time[0]).lpNorm<Eigen::Infinity>() > 1e-12)
        {
            if (world_rank == 0)
                std::cerr << "Error: pulse at initial time is not zero!\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        const BandCoeffs coeffs{alpha, beta, delta, m};
        const HamiltonianOp Hamiltonian{coeffs};
        const VelocityOp velocity_operator{coeffs};

        necessary_functions<Eigen::Matrix2cd, HamiltonianOp, VelocityOp> NF(Hamiltonian, velocity_operator, pulse, Ls,
                                                                                                     Temperature, ef, KBlock, Time, world_rank, world_size);

        NF.ChargeCurrent();

        MPI_Barrier(MPI_COMM_WORLD);

        if (world_rank == 0)
        {
            const auto &TC = NF.TotalChargeCurrent;
            std::ofstream file;

            file.open("./data.csv");
            if (!file.is_open())
            {
                if (world_rank == 0)
                {
                    std::cerr << "Could not open ./data.csv for writing\n";
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            else
            {
                // Results: full round-trip precision. General format, NOT std::fixed --
                // Jy spans ~1e-18..1e-7 and fixed notation would flush the small end to zero.
                file << std::setprecision(17);
                file << "Time,Ax,Ay,Jx,Jy\n";
                for (size_t i = 0; i < Time.size(); ++i)
                {
                    const Eigen::Vector2d A_i = pulse.A(Time[i]);
                    file << Time[i] << "," << A_i[0] << "," << A_i[1] << "," << TC[i][0] << "," << TC[i][1] << "\n";
                }
            }
            file.close();
            std::cout << "MPI run complete; wrote ./data.csv\n";

            file.open("./parameters.txt");
            if (!file.is_open())
            {
                if (world_rank == 0)
                {
                    std::cerr << "Could not open ./parameters.txt for writing\n";
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            else
            {
                file << std::setprecision(6);   // contract: matches format_number(".6g") in Python
                print_core_parameters(file);
                PulseObjectX.printPulseParams(file, "X");
                PulseObjectY.printPulseParams(file, "Y");

                file << "initial time = " << initial_time << "\n";
                file << "final time = " << final_time << "\n";
            }
            file.close();

            std::cout << "\n--- COMPLETING SIMULATION ---" << "\n";
            if (timer)
            {
                timer->stop();
                std::cout << "\nTotal time taken: " << timer->formatted() << "\n";
            }
            printCurrentDateTime();
            std::cout << "\n-----------------------------------------------------------------------------\n";
        }
    }
    catch (const std::exception &e)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Log the error
        std::cerr << "[Rank " << rank << "] FATAL EXCEPTION: \n"
                  << e.what() << std::endl;

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();

    return 0;
}
