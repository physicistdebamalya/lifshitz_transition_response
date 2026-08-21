// main cpp file for tight binding Altermagnetic model

#include <fstream>
#include "necessary_functions.h"
#include "unsupported/Eigen/KroneckerProduct"

// using namespace std;
// using namespace Eigen;

// *************************************************************************

constexpr double V = 1.0;
constexpr double v = 0.5;
constexpr double v3 = 0.5;
constexpr double gamma1 = 0.1;
const std::string System = "Response of Lifshitz transition in a four-band model";
constexpr double Temperature = 0.02;
constexpr double ef = 0.02;
constexpr double dephase = 0.0; // dephasing rate
constexpr double ETP = 100.0;         // extra time percentage: integrate past the pulse by this much
constexpr int TIME_DIVISION = 1000;   // time steps per unit time; sets dt and the whole time grid

constexpr int N1 = 31;
constexpr int N2 = 31;

constexpr double A0X = 0.5;                           // Amplitude of pulse
constexpr double nuX = 0.3;                           // Frequency of pulse
const PulseKind pulse_typeX = PulseKind::Sin2;        // Type of pulse
const PulseKind pulse_wrapperX = PulseKind::Envelope; // "NA" if not and Envelope. "Envelope" if an Envelope of sin pulse
constexpr double sigmaX = 0.1;                        // in case of gaussian pulse is the SD
constexpr double ntt_sigmaX = 1.0;                    // in case of gaussian pulse, total time = ntt_sigma * sigma
constexpr double delayX = 0.0;                        // consider it as initial time t_i, i.e. from 0 how long it delayed to start
constexpr int number_of_cycleX = 10;                  // number of cycles of the pulse
constexpr double durationX = 5;                       // duration of the pulse for linear pulse

constexpr double A0Y = 0.5;
constexpr double nuY = 0.3;
const PulseKind pulse_typeY = PulseKind::Sin2;
const PulseKind pulse_wrapperY = PulseKind::Envelope;
constexpr double sigmaY = 0.1;
constexpr double ntt_sigmaY = 1.0;
constexpr double delayY = 0.0;
constexpr int number_of_cycleY = 10;
constexpr double durationY = 5; // duration of the pulse for linear pulse

// -------------------------------- Think before changing --------------------------------------------------

const Eigen::Vector2d b1{12.0, 0.0};
const Eigen::Vector2d b2{0.0, 12.0};
const Eigen::Vector2d k_origin{-6.0, -6.0};

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
inline c_d term_v(const c_d &p_minus) { return v * p_minus; }
inline c_d term_v3(const c_d &p_plus) { return v3 * p_plus; }

inline Eigen::Vector2cd D_term_v() { return Eigen::Vector2cd{v, -constants::z * v}; }
inline Eigen::Vector2cd D_term_v3() { return Eigen::Vector2cd{v3, constants::z * v3}; }

Eigen::Matrix4cd Hamiltonian(const Eigen::Vector2d &k, const Eigen::Vector2d &A)
{
    Eigen::Matrix4cd H = Eigen::Matrix4cd::Zero();

    const c_d p_plus = shifted_momentum(k, A);
    const c_d p_minus = std::conj(p_plus);

    H(State0, State0) = V / 2.0;
    H(State1, State1) = V / 2.0;
    H(State2, State2) = -V / 2.0;
    H(State3, State3) = -V / 2.0;

    // v couples (State0,State1) and (State2,State3) identically -- computed
    // once, reused for both pairs, mirrored entries taken by conjugation.
    const c_d v_term = term_v(p_minus);
    H(State0, State1) = v_term;
    H(State1, State0) = std::conj(v_term);
    H(State2, State3) = v_term;
    H(State3, State2) = std::conj(v_term);

    const c_d v3_term = term_v3(p_plus);
    H(State0, State3) = v3_term;
    H(State3, State0) = std::conj(v3_term);

    H(State1, State2) = gamma1;
    H(State2, State1) = gamma1;

    return H;
}

// Every entry below is a compile-time-constant expression of v, v3 and
// constants::z (see D_term_v/D_term_v3 above), so there is nothing left to
// recompute per call. Built once here instead of on every observed
// timestep, for every k, across every thread.
const std::array<Eigen::Matrix4cd, 2> VelocityOperatorConst = [] {
    std::array<Eigen::Matrix4cd, 2> co = {Eigen::Matrix4cd::Zero(), Eigen::Matrix4cd::Zero()};

    const Eigen::Vector2cd Dv = D_term_v();
    co[0](State0, State1) = Dv[0];
    co[0](State1, State0) = std::conj(Dv[0]);
    co[0](State2, State3) = Dv[0];
    co[0](State3, State2) = std::conj(Dv[0]);
    co[1](State0, State1) = Dv[1];
    co[1](State1, State0) = std::conj(Dv[1]);
    co[1](State2, State3) = Dv[1];
    co[1](State3, State2) = std::conj(Dv[1]);

    const Eigen::Vector2cd Dv3 = D_term_v3();
    co[0](State0, State3) = Dv3[0];
    co[0](State3, State0) = std::conj(Dv3[0]);
    co[1](State0, State3) = Dv3[1];
    co[1](State3, State0) = std::conj(Dv3[1]);

    return co;
}();

std::array<Eigen::Matrix4cd, 2> velocity_operator([[maybe_unused]] const Eigen::Vector2d &k, [[maybe_unused]] const Eigen::Vector2d &A)
{
    return VelocityOperatorConst;
}

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

const std::vector<Eigen::Matrix4cd, Eigen::aligned_allocator<Eigen::Matrix4cd>> Ls{
    dephase * L1,
    dephase *L2,
    dephase *L3,
    dephase *L4};

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

        necessary_functions<Eigen::Matrix4cd, decltype(Hamiltonian), decltype(velocity_operator)> NF(Hamiltonian, velocity_operator, pulse, Ls,
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
