// main cpp file for tight binding Altermagnetic model

#include <fstream>
#include "./necessary_functions.h"
#include "unsupported/Eigen/KroneckerProduct"

// using namespace std;
// using namespace Eigen;

// *************************************************************************

constexpr double alpha = 1.0;
constexpr double beta = 0.5;
constexpr double delta = 0.5;
constexpr double m = 0.1;
constexpr double Temperature = 0.02;
constexpr double ef = 0.02;
constexpr double dephase = 0.0; // dephasing rate

constexpr int N1 = 31;
constexpr int N2 = 31;

constexpr double A0X = 0.5;                           // Amplitude of pulse
constexpr double nuX = 0.3;                           // Frequency of pulse
const PulseKind pulse_typeX = PulseKind::Sin2;        // Type of pulse
const PulseKind pulse_wrapperX = PulseKind::Envelope; // "NA" if not and Envelope. "Envelope" if an Envelope of sin pulse
constexpr double sigmaX = 0.1;                        // in case of gaussian pulse is the SD
constexpr double ntt_sigmaX = 1.0;                    // in case of gaussian pulse, total time = ntt_sigma * sigma
constexpr double ETPX = 100.0;                        // Extra time percentage
constexpr double delayX = 0.0;                        // consider it as initial time t_i, i.e. from 0 how long it delayed to start
constexpr int number_of_cycleX = 10;                  // number of cycles of the pulse
constexpr double durationX = 5;                       // duration of the pulse for linear pulse

constexpr double A0Y = 0.5;
constexpr double nuY = 0.3;
const PulseKind pulse_typeY = PulseKind::Sin2;
const PulseKind pulse_wrapperY = PulseKind::Envelope;
constexpr double sigmaY = 0.1;
constexpr double ntt_sigmaY = 1.0;
constexpr double ETPY = 100.0;
constexpr double delayY = 0.0;
constexpr int number_of_cycleY = 10;
constexpr double durationY = 5; // duration of the pulse for linear pulse

const Eigen::Vector2d b1{4.0, 0.0};
const Eigen::Vector2d b2{0.0, 4.0};
const Eigen::Vector2d k_origin{-2.0, -2.0};

Eigen::Matrix2cd Hamiltonian(const Eigen::Vector2d &k, const Eigen::Vector2d &A)
{
    Eigen::Matrix2cd Ham;

    c_d Hr1c1 = m;
    c_d Hr1c2 = alpha * pow(k[0] - constants::e * A[0], 2) - delta - constants::z * beta * (k[1] - constants::e * A[1]);

    c_d Hr2c1 = alpha * pow(k[0] - constants::e * A[0], 2) - delta + constants::z * beta * (k[1] - constants::e * A[1]);
    c_d Hr2c2 = m;

    Ham << Hr1c1, Hr1c2,
        Hr2c1, Hr2c2;
    return Ham;
}

std::array<Eigen::Matrix2cd, 2> velocity_operator(const Eigen::Vector2d &k, const Eigen::Vector2d &A)
{
    std::array<Eigen::Matrix2cd, 2> co;

    c_d cox1c1 = 0.0;
    c_d cox1c2 = 2 * (k[0] - constants::e * A[0]) * alpha;

    c_d cox2c1 = 2 * (k[0] - constants::e * A[0]) * alpha;
    c_d cox2c2 = 0.0;

    co[0] << cox1c1, cox1c2,
        cox2c1, cox2c2;

    c_d coy1c1 = 0.0;
    c_d coy1c2 = -constants::z * beta;

    c_d coy2c1 = constants::z * beta;
    c_d coy2c2 = 0.0;

    co[1] << coy1c1, coy1c2,
        coy2c1, coy2c2;

    return co;
}

// *****************************************************************************

const Eigen::Matrix2cd L1{{c_d(1.0), c_d(0.0)},
                          {c_d(0.0), c_d(0.0)}};

const Eigen::Matrix2cd L2{{c_d(0.0), c_d(0.0)},
                          {c_d(0.0), c_d(1.0)}};

const std::vector<Eigen::Matrix2cd, Eigen::aligned_allocator<Eigen::Matrix2cd>> Ls{dephase * L1, dephase *L2};

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

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        printf("Using %d threads.\n", nthreads);
        std::cout << "calculations started for..." << std::endl;
        std::cout << "**********************************************************************************************************************************" << std::endl;

        std::cout << "alpha = " << alpha << std::endl;
        std::cout << "beta = " << beta << std::endl;
        std::cout << "delta = " << delta << std::endl;
        std::cout << "m = " << m << std::endl;
        std::cout << "Temperature = " << Temperature << std::endl;
        std::cout << "ef = " << ef << std::endl;
        std::cout << "dephase = " << dephase << std::endl;

        std::cout << "N1 = " << N1 << std::endl;
        std::cout << "N2 = " << N2 << std::endl;

        std::cout << "b1[0] = " << b1[0] << ", b1[1] = " << b1[1] << std::endl;
        std::cout << "b2[0] = " << b2[0] << ", b2[1] = " << b2[1] << std::endl;
        std::cout << "k_origin[0] = " << k_origin[0] << ", k_origin[1] = " << k_origin[1] << std::endl;

        std::cout << "A0X = " << A0X << std::endl;
        if (A0X != 0.0)
        {
            std::cout << "ETPX = " << ETPX << std::endl;
            std::cout << "delayX = " << delayX << std::endl;
            if (pulse_wrapperX == PulseKind::Envelope)
            {
                std::cout << "pulse_typeX = " << pulse_typeX << " with Envelope" << std::endl;
                if (pulse_typeX == PulseKind::Gaussian)
                {
                    std::cout << "nuX = " << nuX << std::endl;
                    std::cout << "sigmaX = " << sigmaX << std::endl;
                    std::cout << "ntt_sigmaX = " << ntt_sigmaX << std::endl;
                }
                else if (pulse_typeX == PulseKind::Sin2)
                {
                    std::cout << "nuX = " << nuX << std::endl;
                    std::cout << "number_of_cycleX = " << number_of_cycleX << std::endl;
                }
            }
            else if (pulse_wrapperX == PulseKind::NA)
            {
                std::cout << "pulse_typeX = " << pulse_typeX << " without Envelope" << std::endl;
                if (pulse_typeX == PulseKind::Gaussian)
                {
                    std::cout << "sigmaX = " << sigmaX << std::endl;
                    std::cout << "ntt_sigmaX = " << ntt_sigmaX << std::endl;
                }
                else if (pulse_typeX == PulseKind::Sin2)
                {
                    std::cout << "number_of_cycleX = " << number_of_cycleX << std::endl;
                    std::cout << "nuX = " << nuX << std::endl;
                }
                else if (pulse_typeX == PulseKind::Sin || pulse_typeX == PulseKind::Cos)
                {
                    std::cout << "nuX = " << nuX << std::endl;
                }
                else if (pulse_typeX == PulseKind::Linear)
                {
                    std::cout << "durationX = " << durationX << std::endl;
                }
            }
        }

        std::cout << "A0Y = " << A0Y << std::endl;
        if (A0Y != 0.0)
        {
            std::cout << "ETPY = " << ETPY << std::endl;
            std::cout << "delayY = " << delayY << std::endl;
            if (pulse_wrapperY == PulseKind::Envelope)
            {
                std::cout << "pulse_typeY = " << pulse_typeY << " with Envelope" << std::endl;
                if (pulse_typeY == PulseKind::Gaussian)
                {
                    std::cout << "nuY = " << nuY << std::endl;
                    std::cout << "sigmaY = " << sigmaY << std::endl;
                    std::cout << "ntt_sigmaY = " << ntt_sigmaY << std::endl;
                }
                else if (pulse_typeY == PulseKind::Sin2)
                {
                    std::cout << "nuY = " << nuY << std::endl;
                    std::cout << "number_of_cycleY = " << number_of_cycleY << std::endl;
                }
            }
            else if (pulse_wrapperY == PulseKind::NA)
            {
                std::cout << "pulse_typeY = " << pulse_typeY << " without Envelope" << std::endl;
                if (pulse_typeY == PulseKind::Gaussian)
                {
                    std::cout << "sigmaY = " << sigmaY << std::endl;
                    std::cout << "ntt_sigmaY = " << ntt_sigmaY << std::endl;
                }
                else if (pulse_typeY == PulseKind::Sin2)
                {
                    std::cout << "number_of_cycleY = " << number_of_cycleY << std::endl;
                    std::cout << "nuY = " << nuY << std::endl;
                }
                else if (pulse_typeY == PulseKind::Sin || pulse_typeY == PulseKind::Cos)
                {
                    std::cout << "nuY = " << nuY << std::endl;
                }
                else if (pulse_typeY == PulseKind::Linear)
                {
                    std::cout << "durationY = " << durationY << std::endl;
                }
            }
        }

        std::cout << "**********************************************************************************************************************************" << std::endl;

        printCurrentDateTime();

        std::cout << "**********************************************************************************************************************************" << std::endl;
    }

    ProgramTimer timer;

    try
    {
        const PulseType PulseObjectX(pulse_typeX, pulse_wrapperX, A0X, sigmaX, ntt_sigmaX, nuX, ETPX, delayX, number_of_cycleX, durationX);
        const PulseType PulseObjectY(pulse_typeY, pulse_wrapperY, A0Y, sigmaY, ntt_sigmaY, nuY, ETPY, delayY, number_of_cycleY, durationY);

        const double initial_time = std::min(PulseObjectX.ti, PulseObjectY.ti);
        const double final_time = std::max(PulseObjectX.tf, PulseObjectY.tf);

        const int TIME_DIVISION = 1000;
        const int division = (int)(TIME_DIVISION * (final_time - initial_time)) > TIME_DIVISION ? (int)(TIME_DIVISION * (final_time - initial_time)) : TIME_DIVISION;
        const double dt = (std::abs(final_time - initial_time)) / division;

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
        if (pulse.A(Time[0]).lpNorm<Eigen::Infinity>() > 1e-5)
        {
            if (world_rank == 0)
                std::cerr << "Error: pulse at initial time is not zero!\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        necessary_functions<Eigen::Matrix2cd, decltype(Hamiltonian), decltype(velocity_operator)> NF(Hamiltonian, velocity_operator, pulse, Ls,
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
                file << "Time,Ax,Ay,Jx,Jy\n";
                for (size_t i = 0; i < Time.size(); ++i)
                {
                    file << Time[i] << "," << pulse.A(Time[i])[0] << "," << pulse.A(Time[i])[1] << "," << TC[i][0] << "," << TC[i][1] << "\n";
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
                file << "alpha = " << alpha << std::endl;
                file << "beta = " << beta << std::endl;
                file << "delta = " << delta << std::endl;
                file << "m = " << m << std::endl;
                file << "Temperature = " << Temperature << std::endl;
                file << "ef = " << ef << std::endl;
                file << "dephase = " << dephase << std::endl;

                file << "N1 = " << N1 << std::endl;
                file << "N2 = " << N2 << std::endl;

                file << "b1[0] = " << b1[0] << ", b1[1] = " << b1[1] << std::endl;
                file << "b2[0] = " << b2[0] << ", b2[1] = " << b2[1] << std::endl;
                file << "k_origin[0] = " << k_origin[0] << ", k_origin[1] = " << k_origin[1] << std::endl;

                file << "A0X = " << A0X << std::endl;
                if (A0X != 0.0)
                {
                    file << "ETPX = " << ETPX << std::endl;
                    file << "delayX = " << delayX << std::endl;
                    if (pulse_wrapperX == PulseKind::Envelope)
                    {
                        file << "pulse_typeX = " << pulse_typeX << " with Envelope" << std::endl;
                        if (pulse_typeX == PulseKind::Gaussian)
                        {
                            file << "nuX = " << nuX << std::endl;
                            file << "sigmaX = " << sigmaX << std::endl;
                            file << "ntt_sigmaX = " << ntt_sigmaX << std::endl;
                        }
                        else if (pulse_typeX == PulseKind::Sin2)
                        {
                            file << "nuX = " << nuX << std::endl;
                            file << "number_of_cycleX = " << number_of_cycleX << std::endl;
                        }
                    }
                    else if (pulse_wrapperX == PulseKind::NA)
                    {
                        file << "pulse_typeX = " << pulse_typeX << " without Envelope" << std::endl;
                        if (pulse_typeX == PulseKind::Gaussian)
                        {
                            file << "sigmaX = " << sigmaX << std::endl;
                            file << "ntt_sigmaX = " << ntt_sigmaX << std::endl;
                        }
                        else if (pulse_typeX == PulseKind::Sin2)
                        {
                            file << "number_of_cycleX = " << number_of_cycleX << std::endl;
                            file << "nuX = " << nuX << std::endl;
                        }
                        else if (pulse_typeX == PulseKind::Sin || pulse_typeX == PulseKind::Cos)
                        {
                            file << "nuX = " << nuX << std::endl;
                        }
                        else if (pulse_typeX == PulseKind::Linear)
                        {
                            file << "durationX = " << durationX << std::endl;
                        }
                    }
                }

                file << "A0Y = " << A0Y << std::endl;
                if (A0Y != 0.0)
                {
                    file << "ETPY = " << ETPY << std::endl;
                    file << "delayY = " << delayY << std::endl;
                    if (pulse_wrapperY == PulseKind::Envelope)
                    {
                        file << "pulse_typeY = " << pulse_typeY << " with Envelope" << std::endl;
                        if (pulse_typeY == PulseKind::Gaussian)
                        {
                            file << "nuY = " << nuY << std::endl;
                            file << "sigmaY = " << sigmaY << std::endl;
                            file << "ntt_sigmaY = " << ntt_sigmaY << std::endl;
                        }
                        else if (pulse_typeY == PulseKind::Sin2)
                        {
                            file << "nuY = " << nuY << std::endl;
                            file << "number_of_cycleY = " << number_of_cycleY << std::endl;
                        }
                    }
                    else if (pulse_wrapperY == PulseKind::NA)
                    {
                        file << "pulse_typeY = " << pulse_typeY << " without Envelope" << std::endl;
                        if (pulse_typeY == PulseKind::Gaussian)
                        {
                            file << "sigmaY = " << sigmaY << std::endl;
                            file << "ntt_sigmaY = " << ntt_sigmaY << std::endl;
                        }
                        else if (pulse_typeY == PulseKind::Sin2)
                        {
                            file << "number_of_cycleY = " << number_of_cycleY << std::endl;
                            file << "nuY = " << nuY << std::endl;
                        }
                        else if (pulse_typeY == PulseKind::Sin || pulse_typeY == PulseKind::Cos)
                        {
                            file << "nuY = " << nuY << std::endl;
                        }
                        else if (pulse_typeY == PulseKind::Linear)
                        {
                            file << "durationY = " << durationY << std::endl;
                        }
                    }
                }

                file << "initial time = " << initial_time << std::endl;
                file << "final time = " << final_time << std::endl;

                file.close();
            }
            std::cout << "calculations compleated *********************************************************************************************" << std::endl;
            timer.stop();
            std::cout << "Total time taken: " << timer.formatted() << std::endl;
            printCurrentDateTime();
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
