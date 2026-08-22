// main cpp file for tight binding Altermagnetic model

#include <fstream>
#include "necessary_functions.h"
#include "unsupported/Eigen/KroneckerProduct"

// using namespace std;
// using namespace Eigen;

// *************************************************************************

const std::string System = "Response of Lifshitz transition in a four-band model";

// Every column of task.csv this binary reads. A header cell absent from this
// list is a hard error rather than an ignored column -- see
// require_known_columns(). analysis/models.py:verify_cpp_task_columns()
// parses this list out of the source and asserts it equals
// ModelSchema.task_columns, which is the only thing standing between the two
// sides silently drifting (C++ cannot import the schema).
const std::vector<std::string> KNOWN_COLUMNS = {
    "System", "V", "v", "v3", "gamma1", "Temperature", "ef", "dephase",
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
    double V, v, v3, gamma1;
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
    // four_band point runnable only by a four_band binary. Copying the wrong
    // one, or pointing this at another directory's task.csv, would otherwise
    // produce a complete and plausible data.csv labelled with the wrong
    // System. This is the check that closes that.
    if (row.text("System") != System)
        throw std::runtime_error(
            path + ": the System column says \"" + row.text("System") +
            "\" but this binary is \"" + System + "\".");

    TaskSpec T;
    T.V = row.dbl("V");
    T.v = row.dbl("v");
    T.v3 = row.dbl("v3");
    T.gamma1 = row.dbl("gamma1");
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

// -------------------------------- Think before changing --------------------------------------------------

// Basis indices for the 4-band Hamiltonian. Positional and generic: this
// model has no spin structure, so a label borrowed from a BdG model (Up/Dn)
// would assert an identity the basis doesn't have.
enum Basis
{
    State0,
    State1,
    State2,
    State3
};

// Peierls-shifted momentum, in the p+/p- combination the Hamiltonian is
// written in. Computed once and reused by both Hamiltonian and
// velocity_operator below.
inline c_d shifted_momentum(const Eigen::Vector2d &k, const Eigen::Vector2d &A)
{
    const double px = k[0] - constants::e * A[0];
    const double py = k[1] - constants::e * A[1];
    return c_d(px, py);
}

// The off-diagonal Hamiltonian is two Peierls-linear couplings, v and v3.
// term_v occupies two conjugate-mirrored position pairs, term_v3 occupies
// one. D_term_v / D_term_v3 are their k-gradients -- update a term and its
// derivative together, since velocity_operator is built entirely from the
// D_* functions rather than re-derived independently. Both gradients are
// themselves constant (v, v3 enter linearly in k), which is why
// velocity_operator below does not depend on k or A at all.
// The band coefficients, previously namespace-scope constexpr.
//
// EVERY EXPRESSION BELOW IS CHARACTER-IDENTICAL TO THE CONSTEXPR VERSION.
// Only the storage class changed -- see the same note in two_band.cpp for
// why nothing is hand-hoisted: rearranging floating-point expressions to
// "help" would change their association, and leaving them alone makes "the
// numbers cannot have changed" a matter of reading the diff.
struct BandCoeffs
{
    double V, v, v3, gamma1;

    inline c_d term_v(const c_d &p_minus) const { return v * p_minus; }
    inline c_d term_v3(const c_d &p_plus) const { return v3 * p_plus; }

    inline Eigen::Vector2cd D_term_v() const { return Eigen::Vector2cd{v, -constants::z * v}; }
    inline Eigen::Vector2cd D_term_v3() const { return Eigen::Vector2cd{v3, constants::z * v3}; }
};

// Functors rather than free functions, because the parameters they close over
// are no longer compile-time constants. necessary_functions is templated on
// HFunc/voFunc and only ever calls `Ham(k, A)`, so a class with a const
// operator() substitutes with ZERO changes to the engine.
struct HamiltonianOp
{
    BandCoeffs c;

    Eigen::Matrix4cd operator()(const Eigen::Vector2d &k, const Eigen::Vector2d &A) const
    {
        Eigen::Matrix4cd H = Eigen::Matrix4cd::Zero();

        const c_d p_plus = shifted_momentum(k, A);
        const c_d p_minus = std::conj(p_plus);

        H(State0, State0) = c.V / 2.0;
        H(State1, State1) = c.V / 2.0;
        H(State2, State2) = -c.V / 2.0;
        H(State3, State3) = -c.V / 2.0;

        // v couples (State0,State1) and (State2,State3) identically -- computed
        // once, reused for both pairs, mirrored entries taken by conjugation.
        const c_d v_term = c.term_v(p_minus);
        H(State0, State1) = v_term;
        H(State1, State0) = std::conj(v_term);
        H(State2, State3) = v_term;
        H(State3, State2) = std::conj(v_term);

        const c_d v3_term = c.term_v3(p_plus);
        H(State0, State3) = v3_term;
        H(State3, State0) = std::conj(v3_term);

        H(State1, State2) = c.gamma1;
        H(State2, State1) = c.gamma1;

        return H;
    }
};

// Every entry is an expression of v, v3 and constants::z alone (see
// D_term_v/D_term_v3 above) -- no k, no A -- so there is nothing to
// recompute per call. Built ONCE in the constructor, exactly as it used to
// be built once at namespace scope, instead of on every observed timestep,
// for every k, across every thread.
struct VelocityOp
{
    std::array<Eigen::Matrix4cd, 2> co;

    explicit VelocityOp(const BandCoeffs &c)
        : co{Eigen::Matrix4cd::Zero(), Eigen::Matrix4cd::Zero()}
    {
        const Eigen::Vector2cd Dv = c.D_term_v();
        co[0](State0, State1) = Dv[0];
        co[0](State1, State0) = std::conj(Dv[0]);
        co[0](State2, State3) = Dv[0];
        co[0](State3, State2) = std::conj(Dv[0]);
        co[1](State0, State1) = Dv[1];
        co[1](State1, State0) = std::conj(Dv[1]);
        co[1](State2, State3) = Dv[1];
        co[1](State3, State2) = std::conj(Dv[1]);

        const Eigen::Vector2cd Dv3 = c.D_term_v3();
        co[0](State0, State3) = Dv3[0];
        co[0](State3, State0) = std::conj(Dv3[0]);
        co[1](State0, State3) = Dv3[1];
        co[1](State3, State0) = std::conj(Dv3[1]);
    }

    const std::array<Eigen::Matrix4cd, 2> &
    operator()([[maybe_unused]] const Eigen::Vector2d &k,
               [[maybe_unused]] const Eigen::Vector2d &A) const
    {
        return co;
    }
};

// *****************************************************************************

const Eigen::Matrix4cd L1{
    {c_d(1.0), c_d(0.0), c_d(0.0), c_d(0.0)},
    {c_d(0.0), c_d(0.0), c_d(0.0), c_d(0.0)},
    {c_d(0.0), c_d(0.0), c_d(0.0), c_d(0.0)},
    {c_d(0.0), c_d(0.0), c_d(0.0), c_d(0.0)}};

const Eigen::Matrix4cd L2{
    {c_d(0.0), c_d(0.0), c_d(0.0), c_d(0.0)},
    {c_d(0.0), c_d(1.0), c_d(0.0), c_d(0.0)},
    {c_d(0.0), c_d(0.0), c_d(0.0), c_d(0.0)},
    {c_d(0.0), c_d(0.0), c_d(0.0), c_d(0.0)}};

const Eigen::Matrix4cd L3{
    {c_d(0.0), c_d(0.0), c_d(0.0), c_d(0.0)},
    {c_d(0.0), c_d(0.0), c_d(0.0), c_d(0.0)},
    {c_d(0.0), c_d(0.0), c_d(1.0), c_d(0.0)},
    {c_d(0.0), c_d(0.0), c_d(0.0), c_d(0.0)}};

const Eigen::Matrix4cd L4{
    {c_d(0.0), c_d(0.0), c_d(0.0), c_d(0.0)},
    {c_d(0.0), c_d(0.0), c_d(0.0), c_d(0.0)},
    {c_d(0.0), c_d(0.0), c_d(0.0), c_d(0.0)},
    {c_d(0.0), c_d(0.0), c_d(0.0), c_d(1.0)}};

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
    const double V = T.V, v = T.v, v3 = T.v3, gamma1 = T.gamma1;
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

    const std::vector<Eigen::Matrix4cd, Eigen::aligned_allocator<Eigen::Matrix4cd>> Ls{
        dephase * L1,
        dephase * L2,
        dephase * L3,
        dephase * L4};

    auto print_core_parameters = [&](std::ostream &os)
    {
        os << System << "\n";
        os << "V = " << V << "\n";
        os << "v = " << v << "\n";
        os << "v3 = " << v3 << "\n";
        os << "gamma1 = " << gamma1 << "\n";
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
        if (System != "Response of Lifshitz transition in a four-band model")
        {
            std::cerr << "System mismatch.\n It should be: ** Response of Lifshitz transition in a four-band model **\n";
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

        const BandCoeffs coeffs{V, v, v3, gamma1};
        const HamiltonianOp Hamiltonian{coeffs};
        const VelocityOp velocity_operator{coeffs};

        necessary_functions<Eigen::Matrix4cd, HamiltonianOp, VelocityOp> NF(Hamiltonian, velocity_operator, pulse, Ls,
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
