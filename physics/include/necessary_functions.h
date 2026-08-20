// necessary_functions.h

#pragma once
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <string>
#include <deque>
#include <numeric>
#include <chrono>
#include <iomanip>
#include "Eigen/Dense"
#include "unsupported/Eigen/MatrixFunctions"
#include "boost/numeric/odeint.hpp"
#include "boost/numeric/odeint/external/eigen/eigen.hpp"
#include "mpi.h"

#ifdef _OPENMP
inline int max_threads() { return omp_get_max_threads(); }
inline int thread_id() { return omp_get_thread_num(); }
#else
inline int max_threads() { return 1; }
inline int thread_id() { return 0; }
#endif

using c_d = std::complex<double>;

namespace Pauli
{
    // 2×2 complex Pauli matrices
    extern const Eigen::Matrix2cd sx;
    extern const Eigen::Matrix2cd sy;
    extern const Eigen::Matrix2cd sz;
}

namespace constants
{
    inline constexpr c_d z{0.0, 1.0};
    inline constexpr double PI = 3.141592653589793;
    inline constexpr double e = 1.0;
}

class ProgramTimer
{
private:
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;

public:
    double elapsed_time;

    ProgramTimer();
    void stop();
    std::string formatted() const;
};

void printCurrentDateTime();

inline std::string complex_to_string(const c_d value)
{
    std::string result;
    if (value.imag() < 0.0)
    {
        result = std::to_string(value.real()) + "-" + std::to_string(std::abs(value.imag())) + "j";
    }
    else
    {
        result = std::to_string(value.real()) + "+" + std::to_string(std::abs(value.imag())) + "j";
    }
    return result;
}

template <typename DerivedA, typename DerivedB>
auto Commutator(
    const Eigen::MatrixBase<DerivedA> &A,
    const Eigen::MatrixBase<DerivedB> &B)
{
    return ((A.derived() * B.derived()) - (B.derived() * A.derived()));
}

template <typename DerivedA, typename DerivedB>
auto AntiCommutator(
    const Eigen::MatrixBase<DerivedA> &A,
    const Eigen::MatrixBase<DerivedB> &B)
{
    return ((A.derived() * B.derived()) + (B.derived() * A.derived()));
}

enum class PulseKind
{
    Envelope,
    Gaussian,
    Sin2,
    NA,
    Sin,
    Cos,
    Linear
};

inline std::ostream &operator<<(std::ostream &os, PulseKind pulse_kind)
{
    switch (pulse_kind)
    {
    case PulseKind::NA:
        return os << "NA";
    case PulseKind::Envelope:
        return os << "Envelope";
    case PulseKind::Gaussian:
        return os << "Gaussian";
    case PulseKind::Sin2:
        return os << "sin^2";
    case PulseKind::Linear:
        return os << "Linear";
    case PulseKind::Sin:
        return os << "sin";
    case PulseKind::Cos:
        return os << "cos";
    default:
        return os << "Unknown";
    }
}

class PulseType
{
private:
    double A0, sigma, omega, T_pulse, ttotal = 0.0;
    int number_of_cycle_of_pulse = 1;

    PulseKind pulse_type;
    PulseKind pulse_wrapper;

    double gaussian(const double) const;
    double gaussian_envelope(const double) const;
    double sin2(const double) const;
    double sin_envelope(const double) const;
    double Sin(const double) const;
    double Cos(const double) const;

public:
    PulseType() {};
    // full constructor
    PulseType(const PulseKind pulse_type, const PulseKind pulse_wrapper, const double A0, const double sigma, const double ntt_sigma,
              const double nu, const double ETP, const double ti, const int number_of_cycle_of_pulse, const double duration);

    double PULSE(const double) const;

    double ti, tf;
};

class D2Pulse
{
private:
    const PulseType pulseX, pulseY;

public:
    D2Pulse() {};
    D2Pulse(const PulseType &pulseX, const PulseType &pulseY) : pulseX(pulseX), pulseY(pulseY) {};
    inline Eigen::Vector2d A(double t) const
    {
        Eigen::Vector2d pulse;
        pulse[0] = pulseX.PULSE(t);
        pulse[1] = pulseY.PULSE(t);
        return pulse;
    }
};

template <typename matrixType, typename HFunc>
class LMEEvolve
{
private:
    const HFunc &Ham;
    const D2Pulse &pulse;
    const Eigen::Vector2d &k;
    const std::vector<typename matrixType::PlainObject, Eigen::aligned_allocator<typename matrixType::PlainObject>> &Ls;

public:
    LMEEvolve(const HFunc &H_, const D2Pulse &pulse_, const Eigen::Vector2d &k_, const std::vector<typename matrixType::PlainObject, Eigen::aligned_allocator<typename matrixType::PlainObject>> &Ls_)
        : Ham(H_), pulse(pulse_), k(k_), Ls(Ls_) {}

    void operator()(const Eigen::MatrixBase<matrixType> &rho, typename matrixType::PlainObject &drhodt, const double t) const
    {
        const auto Ht = Ham(k, pulse.A(t));
        typename matrixType::PlainObject D = matrixType::PlainObject::Zero();
        for (const auto &L : Ls)
        {
            D.noalias() += L * rho * L.adjoint();
            D.noalias() -= 0.5 * AntiCommutator((L.adjoint() * L), rho);
        }

        drhodt.noalias() = -constants::z * Commutator<matrixType>(Ht, rho) + D;
    }
};

template <typename matrixType, typename voFunc>
class integration_observer
{
private:
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &J_thread;
    const voFunc &VO;
    const D2Pulse &pulse;
    const Eigen::Vector2d &k;
    size_t time_index;

public:
    integration_observer(std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &J_thread_,
                         const voFunc &VO_,
                         const D2Pulse &pulse_,
                         const Eigen::Vector2d &k)
        : J_thread(J_thread_), VO(VO_), pulse(pulse_), k(k), time_index(0) {}

    void operator()(const Eigen::MatrixBase<matrixType> &rho, double t)
    {
        const auto v_ops = VO(k, pulse.A(t)); // Expects std::array or vector return
        const auto rhoT = rho.transpose();

        double jx = constants::e * v_ops[0].cwiseProduct(rhoT).sum().real();
        double jy = constants::e * v_ops[1].cwiseProduct(rhoT).sum().real();

        J_thread[time_index][0] += jx;
        J_thread[time_index][1] += jy;

        ++time_index;
    }
};

template <typename matrixType, typename HFunc, typename voFunc, typename DensityInitFunc>
void integrate_single_k(const Eigen::Vector2d &k,
                        const HFunc &Ham,
                        const voFunc &VO,
                        const D2Pulse &pulse,
                        const std::vector<typename matrixType::PlainObject, Eigen::aligned_allocator<typename matrixType::PlainObject>> &Ls,
                        const std::vector<double> &Time,
                        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &J_thread,
                        DensityInitFunc density_init)
{
    // CRITICAL: Must use vector_space_algebra for Eigen matrices
    // using stepper_type = boost::numeric::odeint::runge_kutta4<
    //     matrixType, double, matrixType, double,
    //     boost::numeric::odeint::vector_space_algebra,
    //     boost::numeric::odeint::default_operations>;
    using stepper_type = boost::numeric::odeint::runge_kutta4<
        typename matrixType::PlainObject, double, typename matrixType::PlainObject, double,
        boost::numeric::odeint::vector_space_algebra,
        boost::numeric::odeint::default_operations>;

    // 1. Initialize Density Matrix
    typename matrixType::PlainObject rho;
    density_init(rho);

    // 2. Setup System
    LMEEvolve<matrixType, HFunc> system(Ham, pulse, k, Ls);
    integration_observer<matrixType, voFunc> obs(J_thread, VO, pulse, k);

    // 3. Time Step
    const double dt = (Time.size() >= 2) ? (Time[1] - Time[0]) : 1e-3;

    // 4. Integrate
    boost::numeric::odeint::integrate_times(
        stepper_type(), system, rho,
        Time.begin(), Time.end(), dt, std::ref(obs));
}

template <typename matrixType, typename HFunc, typename voFunc>
class necessary_functions
{
private:
    static constexpr int M_compileM = matrixType::RowsAtCompileTime;
    static_assert(M_compileM != Eigen::Dynamic, "matrixType must have fixed compile-time dimension\n");

    int world_rank, world_size;

    const HFunc &Ham;
    const voFunc &VO;
    const D2Pulse &pulse;

    const std::vector<typename matrixType::PlainObject, Eigen::aligned_allocator<typename matrixType::PlainObject>> &Ls;

    const double Temperature;
    const double ef;

    const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &KBlock;

    inline double fermi_function(const double x) const
    {
        if (std::abs(Temperature) < 1e-5)
        {
            return ((x > 0.0) ? 0.0 : 1.0);
        }
        const double arg = x / Temperature;

        if (arg > 100.0)
            return 0.0;
        if (arg < -100.0)
            return 1.0;

        return (1.0 / (std::exp(arg) + 1.0));
    }

    const std::vector<double> Time;

    // using stepper_type = boost::numeric::odeint::runge_kutta4<
    //     typename matrixType::PlainObject, double, typename matrixType::PlainObject, double,
    //     boost::numeric::odeint::vector_space_algebra>;

    using stepper_type = boost::numeric::odeint::runge_kutta4<
        typename matrixType::PlainObject, double, typename matrixType::PlainObject, double,
        boost::numeric::odeint::vector_space_algebra,
        boost::numeric::odeint::default_operations>;

    const size_t Nt = Time.size();
    const size_t NK = KBlock.size();

    const size_t chunk = NK / world_size;
    const size_t remainder = NK % world_size;
    const size_t k_start = world_rank * chunk + std::min((size_t)world_rank, remainder);
    const size_t k_end = k_start + chunk + (static_cast<size_t>(world_rank) < remainder ? 1 : 0);

    const size_t local_NK = k_end - k_start;

public:
    mutable std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> TotalChargeCurrent;

    // for all calculations
    necessary_functions(const HFunc &H_, const voFunc &VO_, const D2Pulse &pulse_,
                        const std::vector<typename matrixType::PlainObject, Eigen::aligned_allocator<typename matrixType::PlainObject>> &Ls_, const double Temperature_,
                        const double ef_, const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &KBlock,
                        const std::vector<double> &Time, const int rank_ = 1, const int size_ = 1) : world_rank(rank_), world_size(size_), Ham(H_), VO(VO_), pulse(pulse_), Ls(Ls_), Temperature(Temperature_),
                                                                                                     ef(ef_), KBlock(KBlock), Time(Time)
    {
        TotalChargeCurrent.resize(Time.size(), Eigen::Vector2d::Zero());
    };

    template <typename... Args>
    auto Hamiltonian(Args &&...args) const -> decltype(Ham(std::forward<Args>(args)...))
    {
        return Ham(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto velocity_operator(Args &&...args) const -> decltype(VO(std::forward<Args>(args)...))
    {
        return VO(std::forward<Args>(args)...);
    }

    void density_matrix(const Eigen::Vector2d &, typename matrixType::PlainObject &) const;

    // void LMEEvolve(const EvolveParams &scf = EvolveParams());
    void ChargeCurrent() const;
};

template <typename matrixType, typename HFunc, typename voFunc>
void necessary_functions<matrixType, HFunc, voFunc>::density_matrix(const Eigen::Vector2d &k, typename matrixType::PlainObject &InitialRho) const
{
    static thread_local Eigen::SelfAdjointEigenSolver<matrixType> es;
    Eigen::Matrix<double, M_compileM, 1> Evalues;
    typename matrixType::PlainObject Evectors;

    std::array<int, M_compileM> idx;
    typename matrixType::PlainObject rho_diag = matrixType::PlainObject::Zero();

    const auto mat = Hamiltonian(k, Eigen::Vector2d::Zero());
    es.compute(mat);
    const auto &evals = es.eigenvalues();
    const auto &evecs = es.eigenvectors();
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j)
              { return evals[i] < evals[j]; });

    for (int i = 0; i < M_compileM; i++)
    {
        Evalues[i] = evals.coeff(idx[i]);
        Evectors.col(i) = evecs.col(idx[i]);
    }

    // rho_diag.setZero();

    for (int i = 0; i < M_compileM; ++i)
    {
        rho_diag(i, i) = fermi_function(Evalues[i] - ef);
    }
    InitialRho = Evectors * rho_diag * Evectors.adjoint();
}

template <typename matrixType, typename HFunc, typename voFunc>
void necessary_functions<matrixType, HFunc, voFunc>::ChargeCurrent() const
{
    std::vector<double> local_J_flat(Nt * 2, 0.0);

#pragma omp parallel
    {
        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> J_thread(Nt, Eigen::Vector2d::Zero());

#pragma omp for schedule(dynamic)
        for (size_t k_idx = k_start; k_idx < k_end; ++k_idx)
        {
            const Eigen::Vector2d &k = KBlock[k_idx];

            // Lambda to bridge the gap between integrate_single_k and member function
            // Captures 'this' to access density_matrix() and Temperature
            auto init_func = [&](typename matrixType::PlainObject &rho)
            {
                this->density_matrix(k, rho);
            };

            integrate_single_k<matrixType, HFunc, voFunc, decltype(init_func)>(k, Ham, VO, pulse, Ls, Time, J_thread, init_func);
        }

// Reduce Thread Results -> Node Buffer (Critical Section)
#pragma omp critical
        {
            for (size_t t = 0; t < Nt; ++t)
            {
                local_J_flat[2 * t + 0] += J_thread[t][0];
                local_J_flat[2 * t + 1] += J_thread[t][1];
            }
        }
    } // End OpenMP

    std::vector<double> global_J_flat;
    if (world_rank == 0)
    {
        global_J_flat.resize(Nt * 2);
    }

    MPI_Reduce(local_J_flat.data(),
               (world_rank == 0 ? global_J_flat.data() : nullptr),
               Nt * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        const double invNk = 1.0 / static_cast<double>(NK);
        for (size_t t = 0; t < Nt; ++t)
        {
            TotalChargeCurrent[t] = Eigen::Vector2d(global_J_flat[2 * t], global_J_flat[2 * t + 1]) * invNk;
        }
    }
}
