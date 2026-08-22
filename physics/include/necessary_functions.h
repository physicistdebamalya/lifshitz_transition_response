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

// The exact inverse of operator<< above. Any spelling operator<< would not
// itself print is a hard error naming the valid ones -- so a typo in a
// task.csv cell (`sin2` for `sin^2`) stops the run instead of silently
// selecting a default pulse shape.
PulseKind pulse_kind_from_string(const std::string &text);

// ─────────────────────────────────────────────────────── task.csv reading ──
//
// One parameter point per file, one file per run directory. Parameters used
// to be compile-time constexpr, rewritten into this source per point and
// recompiled inside the run directory; that cannot work on param, where
// `module load gcc/12.4` inside a SLURM job cannot find features.h
// (verified 2026-08-22 on cn001; the same test succeeds on hpc's cn17).
//
// The format is deliberately the simplest thing that can carry a row: a
// header line, one data line, comma-separated, no quoting and no escapes.
// Cells are therefore forbidden from containing a comma, which the writer
// (workflow/task_schema.py) enforces on its side and splitCSV's column-count
// check catches on this one.

std::string trim_cell(const std::string &text);
std::vector<std::string> split_csv(const std::string &line);

// A header cell this binary does not read would otherwise be IGNORED: the
// parameter would keep whatever the loader left in that field and the run
// would produce entirely plausible numbers for a point nobody chose. That is
// the most dangerous failure mode this format introduces, so it is a hard
// error naming the offending cell and listing what was expected.
void require_known_columns(const std::string &path,
                           const std::vector<std::string> &header,
                           const std::vector<std::string> &known);

// Every column is required. Unlike a hand-written task file, this one is
// always produced by exactly one function (task_schema.write_task_csv), so a
// missing column is always a bug -- and a default is the worst possible
// response to a bug, because it produces numbers for a point nobody asked
// for. There are no defaults anywhere in this path.
void require_columns(const std::string &path,
                     const std::vector<std::string> &header,
                     const std::vector<std::string> &required);

// One data row, addressed by column NAME so header order is irrelevant.
class TaskRow
{
public:
    TaskRow(std::string path, std::vector<std::string> header,
            std::vector<std::string> cells);

    // dbl() and integer() require the WHOLE cell to be consumed. Left to
    // themselves std::stod("1.0junk") returns 1.0 and std::stoi("10.5")
    // returns 10, both silently -- exactly the class of quiet wrong answer
    // this layer exists to prevent.
    double dbl(const std::string &key) const;
    int integer(const std::string &key) const;
    std::string text(const std::string &key) const;
    PulseKind kind(const std::string &key) const;

private:
    const std::string &cell(const std::string &key) const;
    std::string path_;
    std::vector<std::string> header_, cells_;
};

// Opens `path`, skips blank and '#' lines, validates the header against
// `known`, and requires EXACTLY ONE data row.
TaskRow read_single_task_row(const std::string &path,
                             const std::vector<std::string> &known);

class PulseType
{
private:
    double A0, sigma, omega, T_pulse, ttotal = 0.0;
    // nu, ntt_sigma and duration were previously constructor-only: used to
    // derive omega/T_pulse/ttotal and then discarded. Kept as members now so
    // printPulseParams() below can report the value the user actually gave,
    // not a derived one -- matching Superconductivity's PulseType, which
    // stores the same three for the same reason.
    double nu = 0.0, ntt_sigma = 0.0, duration = 0.0;
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

    // Category-C pulse-parameter emission (see analysis/snippets.py's
    // three-category rule this mirrors): writes A0<axis> unconditionally,
    // then only the fields this pulse's own shape actually uses. One
    // definition, shared by every model's print_core_parameters instead of
    // duplicated per model -- see necessary_functions.cpp for the exact
    // conditional structure, ported from Superconductivity's
    // PulseType::printPulseParams (necessary_functions.cpp there), minus the
    // phi/carrier-phase handling that pulse model has and this one does not.
    void printPulseParams(std::ostream &, const std::string &) const;

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

// A single-slot memo in front of D2Pulse::A.
//
// Per accepted timestep, odeint's runge_kutta4 evaluates the system at t,
// t + dt/2 (twice, k2 and k3 share one `th`) and t + dt, and integrate_times
// calls the observer at t as well. That is five A(t) evaluations landing on
// three distinct times, with the observer's t always repeating the step's
// first stage. Remembering the most recent (t, A) pair collapses the five to
// two. It is worth doing because each A(t) is two PulseType::PULSE calls --
// a nested switch plus exp/sin -- run for every k-point of every timestep.
//
// Deliberately not shared. One cache is constructed per k-point on the stack
// of the OpenMP worker that owns that k (see integrate_single_k), so the
// mutable state is never reachable from a second thread.
//
// The lookup compares t bit-exactly, so a hit returns precisely the value
// D2Pulse::A would have recomputed. No result changes.
class PulseCache
{
private:
    const D2Pulse &pulse;
    mutable Eigen::Vector2d cached_A = Eigen::Vector2d::Zero();
    mutable double cached_t = 0.0;
    mutable bool valid = false;

public:
    explicit PulseCache(const D2Pulse &pulse_) : pulse(pulse_) {}

    const Eigen::Vector2d &A(const double t) const
    {
        if (!valid || t != cached_t)
        {
            cached_A = pulse.A(t);
            cached_t = t;
            valid = true;
        }
        return cached_A;
    }
};

template <typename matrixType, typename HFunc>
class LMEEvolve
{
private:
    const HFunc &Ham;
    const PulseCache &pulse;
    const Eigen::Vector2d &k;
    const std::vector<typename matrixType::PlainObject, Eigen::aligned_allocator<typename matrixType::PlainObject>> &Ls;

public:
    LMEEvolve(const HFunc &H_, const PulseCache &pulse_, const Eigen::Vector2d &k_, const std::vector<typename matrixType::PlainObject, Eigen::aligned_allocator<typename matrixType::PlainObject>> &Ls_)
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
    const PulseCache &pulse;
    const Eigen::Vector2d &k;
    size_t time_index;

public:
    integration_observer(std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &J_thread_,
                         const voFunc &VO_,
                         const PulseCache &pulse_,
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
    // One cache, shared by the stepper and the observer, so the vector
    // potential at a given t is computed once and reused by both. Stack-local
    // to this k-point, hence private to the calling thread.
    const PulseCache pulse_cache(pulse);

    LMEEvolve<matrixType, HFunc> system(Ham, pulse_cache, k, Ls);
    integration_observer<matrixType, voFunc> obs(J_thread, VO, pulse_cache, k);

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
