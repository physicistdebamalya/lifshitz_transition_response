#include "Eigen/Dense"
#include "boost/numeric/odeint.hpp"
#include "boost/archive/binary_oarchive.hpp"
#include "boost/archive/binary_iarchive.hpp"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_roots.h"
#include <omp.h>
#include <fstream>
#include "pulse_type.h"
// #include "boost_serialization_eigen.h"
// #include "eigen_boost_serialization.hpp"

using namespace std;
using namespace Eigen;
using namespace boost::numeric::odeint;

ostringstream oss;

const double eps_abs = 1e-8;
const double eps_rel = 1e-6;
const int precision = 10;
void setOssPrecision()
{
    // Clear the contents and reset the state of the global ostringstream object
    oss.str("");
    oss.clear();

    // Set the precision of the global ostringstream object to 10 decimal places
    oss << setprecision(precision);
}

void printCurrentDateTime()
{
    // Get current time as a time_point object
    auto now = chrono::system_clock::now();

    // Convert time_point object to time_t
    time_t currentTime = chrono::system_clock::to_time_t(now);

    // Convert time_t to tm struct
    tm *timeStruct = localtime(&currentTime);

    // Print date and time
    cout << "date: " << put_time(timeStruct, "%d-%m-%Y") << ", time: " << put_time(timeStruct, "%H-%M-%S") << endl;
}

typedef complex<double> c_d;
typedef Vector4cd state_type;
typedef Vector2cd state_type_Schrodinger;

// bulirsch_stoer_dense_out<state_type_Schrodinger> stepper_type_Schrodinger(eps_abs, eps_rel);

typedef runge_kutta_dopri5<state_type_Schrodinger> stepper_type_Schrodinger;
typedef runge_kutta_dopri5<state_type> stepper_type;

// typedef runge_kutta_cash_karp54<state_type_Schrodinger> stepper_type_Schrodinger;
// typedef runge_kutta_cash_karp54<state_type> stepper_type;

const c_d z(0, 1);
const Matrix2cd PauliMatrix_1{{0, 1}, {1, 0}};
const Matrix2cd PauliMatrix_2{{0, -z}, {z, 0}};
const Matrix2cd PauliMatrix_3{{1, 0}, {0, -1}};
const Matrix2cd Identity_2{{1.0, 0}, {0, 1.0}};

const double hbar = 1;
const double t1 = 1;
const double t2 = 1;
const double M = 0.1;
const double a = 1;
const double q = 1;
const double delta_x = 1;

const double ef = 1.00498756;
const double dephase = 0;

// Nx and Ny should be multiple of 300 in this case. This can be modified in the integration function if
// needed of memory management
const int Nx = 15;
const int Ny = 15;
const double lower_limitX = -2;
const double upper_limitX = 2;
const double lower_limitY = -2;
const double upper_limitY = 2;
const double extra_time_percentage = 100;

// x component
const string pulse_type_x = "sin^2";
const double A0E_x = 0.01;
const string pulse_wrapper_x = "envelope";
const double A0_x = 0.01;
const double nu_x = 0.3;
const int number_of_cycle_x = 10;
const double phi_x = 0;

const double T_x = (1 / nu_x);
const double x_delay = (phi_x / (2 * M_PI * nu_x));

// y component
const string pulse_type_y = "sin^2";
const double A0E_y = 0.01;
const string pulse_wrapper_y = "envelope";
const double A0_y = 0.01;
const double nu_y = 0.3;
const int number_of_cycle_y = 10;
const double phi_y = 0;

const double T_y = (1 / nu_y);
const double y_delay = (phi_y / (2 * M_PI * nu_x));

// gaussian
const double sigma_x = 5.0;
const double sigma_y = 5.0;
const double ntt_sigmaX = 10;
const double ntt_sigmaY = 10;

// sin^2

// box_type
const double nu_e_x = 0.3;
const double nu_e_y = 0.3;
const double ETP_x = 0;
const double ETP_y = 0;

double initial_time, final_time, ti, tf;

pair<double, double> initial_times(const string pulse_type, const double delay, const double number_of_cycle, const double T, const double sigma, const double ntt_sigma, const double nu_e, const double ETP, const double x) // x is the percentage of extra time of total time
{
    double initial_time, final_time;

    initial_time = delay;
    if (pulse_type == "sin^2")
    {
        final_time = initial_time + (number_of_cycle * T);
    }
    else if (pulse_type == "gaussian")
    {
        final_time = initial_time + (ntt_sigma * sigma);
    }
    else if (pulse_type == "box_type")
    {
        final_time = initial_time + (number_of_cycle * T) + (1.0 / (2 * nu_e));
        if (pulse_wrapper_x == "add")
        {
            final_time += ((number_of_cycle * T) * ETP / 100.0);
        }
    }
    else
    {
        throw invalid_argument("The pulse type is not supported.");
    }
    final_time += (abs(final_time - initial_time) * (x / 100.0));
    return make_pair(initial_time, final_time);
}

void initialization()
{
    pair<double, double> x_times = initial_times(pulse_type_x, x_delay, number_of_cycle_x, T_x, sigma_x, ntt_sigmaX, nu_e_x, ETP_x, extra_time_percentage);
    pair<double, double> y_times = initial_times(pulse_type_y, y_delay, number_of_cycle_y, T_y, sigma_y, ntt_sigmaY, nu_e_y, ETP_y, extra_time_percentage);

    initial_time = min(x_times.first, y_times.first);
    final_time = max(x_times.second, y_times.second);
}

double (PulseType::*A_pulse)(const double) const = &PulseType::PULSE;

PulseType PulseObject_x(pulse_type_x, pulse_wrapper_x, A0E_x, A0_x, sigma_x, ntt_sigmaX, nu_e_x, nu_x, ETP_x, x_delay, number_of_cycle_x);
PulseType PulseObject_y(pulse_type_y, pulse_wrapper_y, A0E_y, A0_y, sigma_y, ntt_sigmaY, nu_e_y, nu_y, ETP_y, y_delay, number_of_cycle_y);
// (pulse_type, pulse_wrapper, A0E, A0, sigma, ntt_sigma, nu_e, nu, ETP, const double ti, const int number_of_cycle_of_pulse)

double Ax(const double t)
{
    return (PulseObject_x.*A_pulse)(t);
}
double Ay(const double t)
{
    return (PulseObject_y.*A_pulse)(t);
}

double m(const double t, const double px, const double py) // changing the functional form will effect the derivative of velocity operator
{
    return M;
}

double sigmax_term(const double t, const double px, const double py)
{
    double sigmax_term = (t1 * pow(((px - (q * Ax(t))) * a) / hbar, 2)) - delta_x;
    return sigmax_term;
}
double sigmay_term(const double t, const double px, const double py)
{
    double sigmay_term = (t2 * (((py - (q * Ay(t))) * a) / hbar));
    return sigmay_term;
}
double sigmaz_term(const double t, const double px, const double py)
{
    double sigmaz_term = m(t, px, py);
    return (sigmaz_term);
}
double identity_term(const double t, const double px, const double py)
{
    double identity_term = 0.0;
    return identity_term;
}

Matrix2cd H(const double t, const double px, const double py)
{
    Matrix2cd Hamiltonian;
    Hamiltonian = (sigmax_term(t, px, py) * PauliMatrix_1) + (sigmay_term(t, px, py) * PauliMatrix_2) + (sigmaz_term(t, px, py) * PauliMatrix_3) + (identity_term(t, px, py) * Identity_2);
    return Hamiltonian;
}

Matrix2cd eigenvector(const double t, const double px, const double py)
{
    ComplexEigenSolver<Matrix2cd> es(H(t, px, py));
    Vector2d vals = es.eigenvalues().real();
    vector<int> idx(vals.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&vals](int i, int j)
         { return vals[i] < vals[j]; });
    Matrix2cd vectors;
    for (int i = 0; i < vals.size(); i++)
    {
        vectors.col(i) = es.eigenvectors().col(idx[i]);
    }
    return vectors;
}

Vector2d eigenvalue(const double t, const double px, const double py)
{
    ComplexEigenSolver<Matrix2cd> es(H(t, px, py));
    Vector2d vals = es.eigenvalues().real();
    vector<int> idx(vals.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&vals](int i, int j)
         { return vals[i] < vals[j]; });
    Vector2d value;
    for (int i = 0; i < vals.size(); i++)
    {
        value[i] = vals.coeff(idx[i]);
    }
    return value;
}

double epsilon(const double t, const double px, const double py)
{
    double epsilon = sqrt(pow(sigmax_term(t, px, py), 2) + pow(sigmay_term(t, px, py), 2) + pow(sigmaz_term(t, px, py), 2)) + identity_term(t, px, py);
    return epsilon;
}

double epsilon_G(const double t, const double px, const double py)
{
    double epsilon = -sqrt(pow(sigmax_term(t, px, py), 2) + pow(sigmay_term(t, px, py), 2) + pow(sigmaz_term(t, px, py), 2)) + identity_term(t, px, py);
    return epsilon;
}

Matrix2cd UnitaryMatrix(const double t, const double px, const double py)
{
    Matrix2cd UM;
    const double X = sigmax_term(t, px, py);
    const double Y = sigmay_term(t, px, py);
    const double Z = sigmaz_term(t, px, py);
    c_d UM11 = -((-Z + sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2))) / (X + z * Y)) / sqrt(1 + pow(Z - sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2)), 2) / (pow(X, 2) + pow(Y, 2)));
    c_d UM12 = -((-Z - sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2))) / (X + z * Y)) / sqrt(1 + pow(Z + sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2)), 2) / (pow(X, 2) + pow(Y, 2)));
    c_d UM21 = 1 / sqrt(1 + pow(Z - sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2)), 2) / (pow(X, 2) + pow(Y, 2)));
    c_d UM22 = 1 / sqrt(1 + pow(Z + sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2)), 2) / (pow(X, 2) + pow(Y, 2)));
    if (isnan(abs(UM11)) || isnan(abs(UM12)) || isnan(abs(UM21)) || isnan(abs(UM22)))
    {
        UM = eigenvector(t, px, py);
    }
    else
    {
        UM(0, 0) = UM11;
        UM(0, 1) = UM12;
        UM(1, 0) = UM21;
        UM(1, 1) = UM22;
    }

    return UM;
}

// Matrix4cd psudo_H(const double t, const double px, const double py)
// {
//     Matrix4cd psudo_H;
//     Matrix2cd H1 = H(t, px, py);
//     psudo_H << 0.0, -H1(1, 0), H1(0, 1), 0.0,
//         -H1(0, 1), H1(0, 0) - H1(1, 1), 0.0, H1(0, 1),
//         H1(1, 0), 0.0, H1(1, 1) - H1(0, 0), -H1(1, 0),
//         0.0, H1(1, 0), -H1(0, 1), 0.0;
//     return psudo_H;
// }

// class MySystem
// {
// private:
//     double px, py;

// public:
//     MySystem(double i, double j) : px(i), py(j) {}
//     void operator()(const state_type &y, state_type &dydt, const double t) const
//     {
//         dydt = -(z / hbar) * (psudo_H(t, px, py) * y);
//     }
// };

Matrix4cd psudo_H(const double t, const double px, const double py)
{
    Matrix4cd psudo_H;
    Matrix2cd H1 = H(t, px, py);
    psudo_H << 0.0, (z / hbar) * H1(1, 0), -(z / hbar) * H1(0, 1), 0.0,
        (z / hbar) * H1(0, 1), -dephase - (z / hbar) * H1(0, 0) + (z / hbar) * H1(1, 1), 0.0, -(z / hbar) * H1(0, 1),
        -(z / hbar) * H1(1, 0), 0.0, -dephase - (z / hbar) * H1(1, 1) + (z / hbar) * H1(0, 0), (z / hbar) * H1(1, 0),
        0.0, -(z / hbar) * H1(1, 0), (z / hbar) * H1(0, 1), 0.0;
    return psudo_H;
}

class MySystem
{
private:
    double px, py;

public:
    MySystem(double i, double j) : px(i), py(j) {}
    void operator()(const state_type &y, state_type &dydt, const double t) const
    {
        dydt = (psudo_H(t, px, py) * y);
    }
};

double Dxsigma_x(const double t, const double px, const double py)
{
    double Dxsigma_x = 2 * pow(a, 2) * t1 * ((px - (q * Ax(t))) / pow(hbar, 2));
    return Dxsigma_x;
}
double Dxsigma_y(const double t, const double px, const double py)
{
    double Dxsigma_y = 0.0;
    return Dxsigma_y;
}
double Dxsigma_z(const double t, const double px, const double py)
{
    double Dxsigma_z = 0.0;
    return (Dxsigma_z);
}
double Dxidentity(const double t, const double px, const double py)
{
    double Dxidentity = 0.0;
    return Dxidentity;
}

double Dysigma_x(const double t, const double px, const double py)
{
    double Dysigma_x = 0.0;
    return Dysigma_x;
}
double Dysigma_y(const double t, const double px, const double py)
{
    double Dysigma_y = t2 * (a / hbar);
    return Dysigma_y;
}
double Dysigma_z(const double t, const double px, const double py)
{
    double Dysigma_z = 0.0;
    return (Dysigma_z);
}
double Dyidentity(const double t, const double px, const double py)
{
    double Dyidentity = 0.0;
    return Dyidentity;
}

double Dxepsilon(const double t, const double px, const double py)
{
    const double dIx = Dxidentity(t, px, py);
    const double Y = sigmay_term(t, px, py);
    const double dxx = Dxsigma_x(t, px, py);
    const double X = sigmax_term(t, px, py);
    const double dxy = Dxsigma_y(t, px, py);
    const double Z = sigmaz_term(t, px, py);
    const double dxz = Dxsigma_z(t, px, py);

    double Dxepsilon = dIx + (((X * dxx) + (Y * dxy) + (Z * dxz)) / (sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2))));
    return Dxepsilon;
}

double Dyepsilon(const double t, const double px, const double py)
{
    const double dIy = Dyidentity(t, px, py);
    const double dyz = Dysigma_z(t, px, py);
    const double Y = sigmay_term(t, px, py);
    const double X = sigmax_term(t, px, py);
    const double Z = sigmaz_term(t, px, py);
    const double dyy = Dysigma_y(t, px, py);
    const double dyx = Dysigma_x(t, px, py);

    double Dyepsilon = dIy + (((X * dyx) + (Y * dyy) + (Z * dyz)) / (sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2))));
    return Dyepsilon;
}

double Dxepsilon_G(const double t, const double px, const double py)
{
    const double dIx = Dxidentity(t, px, py);
    const double Y = sigmay_term(t, px, py);
    const double dxx = Dxsigma_x(t, px, py);
    const double X = sigmax_term(t, px, py);
    const double dxy = Dxsigma_y(t, px, py);
    const double Z = sigmaz_term(t, px, py);
    const double dxz = Dxsigma_z(t, px, py);

    double Dxepsilon = dIx - (((X * dxx) + (Y * dxy) + (Z * dxz)) / (sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2))));
    return Dxepsilon;
}

double Dyepsilon_G(const double t, const double px, const double py)
{
    const double dIy = Dyidentity(t, px, py);
    const double dyz = Dysigma_z(t, px, py);
    const double Y = sigmay_term(t, px, py);
    const double X = sigmax_term(t, px, py);
    const double Z = sigmaz_term(t, px, py);
    const double dyy = Dysigma_y(t, px, py);
    const double dyx = Dysigma_x(t, px, py);

    double Dyepsilon = dIy - (((X * dyx) + (Y * dyy) + (Z * dyz)) / (sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2))));
    return Dyepsilon;
}

Matrix2cd jx(const double t, const double px, const double py)
{
    Matrix2cd current = q * ((Dxsigma_x(t, px, py) * PauliMatrix_1) + (Dxsigma_y(t, px, py) * PauliMatrix_2) + (Dxsigma_z(t, px, py) * PauliMatrix_3) + (Dxidentity(t, px, py) * Identity_2));
    return current;
}
Matrix2cd jy(const double t, const double px, const double py)
{
    Matrix2cd current = q * ((Dysigma_x(t, px, py) * PauliMatrix_1) + (Dysigma_y(t, px, py) * PauliMatrix_2) + (Dysigma_z(t, px, py) * PauliMatrix_3) + (Dyidentity(t, px, py) * Identity_2));
    return current;
}

Matrix2cd Sx(const double t, const double px, const double py)
{
    Matrix2cd sx = PauliMatrix_1;
    return sx;
}
Matrix2cd Sy(const double t, const double px, const double py)
{
    Matrix2cd sy = PauliMatrix_2;
    return sy;
}

double fermi_function(const double x)
{
    double theta_function;
    if (x > 0)
    {
        theta_function = 0.0;
    }
    else
    {
        theta_function = 1.0;
    }
    return theta_function;
}

Matrix2cd InitialRho(const double t, const double px, const double py)
{
    Matrix2cd UM = UnitaryMatrix(t, px, py);
    Matrix2cd ireb{{fermi_function(epsilon_G(t, px, py) - ef) * sqrt(1.0 + pow(Dxepsilon_G(t, px, py), 2) + pow(Dyepsilon_G(t, px, py), 2)), 0.0},
                   {0.0, fermi_function(epsilon(t, px, py) - ef) * sqrt(1.0 + pow(Dxepsilon(t, px, py), 2) + pow(Dyepsilon(t, px, py), 2))}};
    Matrix2cd irab = UM * ireb * UM.adjoint();
    return irab;
}

class observer
{
private:
    vector<Matrix2cd> &I;

public:
    observer(vector<Matrix2cd> &reference_i) : I(reference_i) {};
    void operator()(const state_type &y, const double t)
    {
        state_type psudo_psi = y;
        Matrix2cd RHO{{psudo_psi[0], psudo_psi[1]},
                      {psudo_psi[2], psudo_psi[3]}};

        I.push_back(RHO);
    }
};

class integrate_MySystem
{
private:
    double px, py;
    vector<double> time_range;
    state_type y;

public:
    vector<Matrix2cd> I;
    integrate_MySystem() {} // Default constructor
    integrate_MySystem(vector<double> i, double j, double k) : time_range(i), px(j), py(k)
    {
        I.reserve(time_range.size());
        const double dt = 0.001;
        MySystem LVNE(px, py);
        Matrix2cd y0 = InitialRho(ti, px, py);
        y << y0(0, 0), y0(0, 1),
            y0(1, 0), y0(1, 1);

        integrate_times(make_dense_output(eps_abs, eps_rel, stepper_type()), LVNE, y, time_range, dt, observer(I));
    }
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & I;
    }
};

double SurfaceIntegration(const double t)
{
    if ((lower_limitX >= upper_limitX) || (lower_limitY >= upper_limitY))
    {
        throw domain_error("in integration check lower and upper limit of x and y carefully");
    }
    const double h_x = abs(lower_limitX - upper_limitX) / Nx;
    const double h_y = abs(lower_limitY - upper_limitY) / Ny;
    double weight_x;
    double weight_y;
    double weight;
    double sumG = 0;
    double sumE = 0;
    for (int i = 0; i <= Nx; i++)
    {
        for (int j = 0; j <= Ny; j++)
        {
            const double px = lower_limitX + (h_x * i);
            const double py = lower_limitY + (h_y * j);
            double SurfaceElementG = sqrt(1.0 + pow(Dxepsilon_G(t, px, py), 2) + pow(Dyepsilon_G(t, px, py), 2)) * fermi_function(epsilon_G(t, px, py) - ef);
            double SurfaceElementE = sqrt(1.0 + pow(Dxepsilon(t, px, py), 2) + pow(Dyepsilon(t, px, py), 2)) * fermi_function(epsilon(t, px, py) - ef);

            if (isnan(SurfaceElementG) || isnan(SurfaceElementE))
            {
                throw runtime_error("NAN value appeared at in integration at \n"
                                    "time = " +
                                    to_string(t) + "\n"
                                                   "px = " +
                                    to_string(px) + "\n"
                                                    "py = " +
                                    to_string(py));
            }

            if (i == 0 || i == Nx)
            {
                weight_x = 1;
            }
            else if (i % 3 == 0)
            {
                weight_x = 2;
            }
            else
            {
                weight_x = 3;
            }
            if (j == 0 || j == Ny)
            {
                weight_y = 1;
            }
            else if (j % 3 == 0)
            {
                weight_y = 2;
            }
            else
            {
                weight_y = 3;
            }
            weight = (9.0 / 64.0) * h_x * h_y * weight_x * weight_y * sin(M_PI / 2.0);
            sumG += weight * SurfaceElementG;
            sumE += weight * SurfaceElementE;
        }
    }
    return (sumG + sumE);
}

void integration(const double lower_limitX, const double upper_limitX, const double lower_limitY, const double upper_limitY, const vector<double> time, Matrix2cd (*func1)(double, double, double), Matrix2cd (*func2)(double, double, double), vector<double> &integration_vals1, vector<double> &integration_vals2)
{
    atomic<bool> stopExecution(false);
    exception_ptr excPtr = nullptr;
    if ((lower_limitX >= upper_limitX) || (lower_limitY >= upper_limitY))
    {
        throw domain_error("in integration check lower and upper limit of x and y carefully");
    }
    const int Nx = 300;
    const int Ny = 300;

    vector<vector<integrate_MySystem>> objects(Nx + 1, vector<integrate_MySystem>(Ny + 1));
    const double h_x = abs(lower_limitX - upper_limitX) / Nx;
    const double h_y = abs(lower_limitY - upper_limitY) / Ny;
#pragma omp parallel for collapse(2)
    for (int i = 0; i <= Nx; i++)
    {
        for (int j = 0; j <= Ny; j++)
        {
            if (stopExecution)
            {
                continue;
            }
            try
            {
                double px = lower_limitX + (h_x * i);
                double py = lower_limitY + (h_y * j);
                objects[i][j] = integrate_MySystem(time, px, py);
            }
            catch (...)
            {
#pragma omp critical
                {
                    if (!excPtr) // Only store the first exception we catch
                    {
                        excPtr = current_exception();
                    }
                    stopExecution = true;
                }
                continue;
            }
        }
    }

#pragma omp parallel for
    for (int time_point = 0; time_point < time.size(); time_point++)
    {
        if (stopExecution)
        {
            continue;
        }
        try
        {
            double weight_x;
            double weight_y;
            double weight;
            double sum1 = 0;
            double sum2 = 0;
            for (int i = 0; i <= Nx; i++)
            {
                for (int j = 0; j <= Ny; j++)
                {
                    const double px = lower_limitX + (h_x * i);
                    const double py = lower_limitY + (h_y * j);
                    Matrix2cd Rho = objects[i][j].I[time_point];
                    double func_value1 = ((func1(time[time_point], px, py) * Rho).trace().real());
                    double func_value2 = ((func2(time[time_point], px, py) * Rho).trace().real());

                    if (isnan(func_value1) || isnan(func_value2))
                    {
                        throw runtime_error("NAN value appeared at in integration at \n"
                                            "time = " +
                                            to_string(time[time_point]) + "\n"
                                                                          "px = " +
                                            to_string(px) + "\n"
                                                            "py = " +
                                            to_string(py));
                    }

                    if (i == 0 || i == Nx)
                    {
                        weight_x = 1;
                    }
                    else if (i % 3 == 0)
                    {
                        weight_x = 2;
                    }
                    else
                    {
                        weight_x = 3;
                    }
                    if (j == 0 || j == Ny)
                    {
                        weight_y = 1;
                    }
                    else if (j % 3 == 0)
                    {
                        weight_y = 2;
                    }
                    else
                    {
                        weight_y = 3;
                    }
                    weight = (9.0 / 64.0) * h_x * h_y * weight_x * weight_y * sin(M_PI / 2.0);
                    sum1 += weight * func_value1;
                    sum2 += weight * func_value2;
                }
            }
            integration_vals1[time_point] = sum1;
            integration_vals2[time_point] = sum2;
        }
        catch (...)
        {
#pragma omp critical
            {
                if (!excPtr) // Only store the first exception we catch
                {
                    excPtr = current_exception();
                }
                stopExecution = true;
            }
            continue;
        }
    }
    if (excPtr)
    {
        rethrow_exception(excPtr);
    }
}

// namespace boost
// {
//     namespace serialization
//     {
//         template <class Archive>
//         void serialize(Archive &ar, vector<Matrix<complex<double>, 2, 2>> &v, const unsigned int version)
//         {
//             size_t size = v.size();
//             ar & size; // Serialize the size of the vector
//             if (Archive::is_loading::value)
//             {
//                 v.resize(size); // Resize the vector to the correct size before deserialization
//             }
//             for (auto &matrix : v)
//                 ar & matrix; // Then iteratively save/load the elements
//         }
//         template <class Archive>
//         void serialize(Archive &ar, Matrix<complex<double>, 2, 2> &matrix, const unsigned int version)
//         {
//             if (Archive::is_loading::value)
//             {
//                 matrix.resize(2, 2); // Ensure the matrix is of correct size before deserialization
//             }
//             for (int i = 0; i < 2; ++i)
//                 for (int j = 0; j < 2; ++j)
//                     ar &matrix(i, j);
//         }
//         template <class Archive>
//         void serialize(Archive &ar, complex<double> &c, const unsigned int version)
//         {
//             double real_part = c.real();
//             double imag_part = c.imag();
//             ar & real_part;
//             ar & imag_part;
//             if (Archive::is_loading::value)
//             {
//                 c = complex<double>(real_part, imag_part); // Reconstruct the complex number during deserialization
//             }
//         }

//         template <class Archive>
//         void serialize(Archive &ar, tuple<int, int, integrate_MySystem> &t, const unsigned int version)
//         {
//             ar &get<0>(t);
//             ar &get<1>(t);
//             ar &get<2>(t);
//         }

//     } // namespace serialization
// } // namespace boost

// vector<Matrix2cd> retrieveI(const int i, const int j, const streampos &pos, ifstream &ifs)
// {
//     boost::archive::binary_iarchive ia(ifs);
//     tuple<int, int, integrate_MySystem> obj;
//     ia >> obj;

//     // Found the (i, j) pair in the index file
//     // Jump to the corresponding position in the binary file
//     ifs.seekg(pos, ios::beg);
//     ia >> obj;
//     auto result = get<2>(obj);
//     int objIndexI = get<0>(obj);
//     int objIndexJ = get<1>(obj);
//     if (objIndexI != i || objIndexJ != j)
//     {
//         throw runtime_error("data mismatch in files");
//     }
//     vector<Matrix2cd> I_return = result.I;
//     ifs.clear();
//     ifs.seekg(0, ios::beg);
//     return I_return;
// }

// // *****************************************************
// // For huge Nx and Ny
// void integration(const vector<double> &time, Matrix2cd (*func1)(double, double, double), Matrix2cd (*func2)(double, double, double), vector<double> &integration_vals1, vector<double> &integration_vals2)
// {
//     // *****************************************************
//     // For huge Nx and Ny
//     atomic<bool> stopExecution(false);
//     exception_ptr excPtr = nullptr;
//     if ((lower_limitX >= upper_limitX) || (lower_limitY >= upper_limitY))
//     {
//         throw domain_error("in integration check lower and upper limit of x and y carefully");
//     }
//     ofstream ofs("integrate_MySystem.bin", ios::binary | ios::out);
//     vector<vector<streampos>> objects(Nx + 1, vector<streampos>(Ny + 1));
//     boost::archive::binary_oarchive oa(ofs);
//     integrate_MySystem dummy(time, 0, 0);
//     oa << make_tuple(-1, -1, dummy);

//     const double h_x = abs(lower_limitX - upper_limitX) / Nx;
//     const double h_y = abs(lower_limitY - upper_limitY) / Ny;
//     int count_iteration_calculation = 0;
// #pragma omp parallel for collapse(2)
//     for (int i = 0; i <= Nx; i++)
//     {
//         for (int j = 0; j <= Ny; j++)
//         {
//             if (stopExecution)
//             {
//                 continue;
//             }
//             try
//             {
//                 double px = lower_limitX + (h_x * i);
//                 double py = lower_limitY + (h_y * j);
//                 integrate_MySystem result(time, px, py);
// #pragma omp critical
//                 {
//                     streampos pos = ofs.tellp();
//                     oa << make_tuple(i, j, result);
//                     objects[i][j] = pos;
//                 }
//             }
//             catch (...)
//             {
// #pragma omp critical
//                 {
//                     if (!excPtr) // Only store the first exception we catch
//                     {
//                         excPtr = current_exception();
//                     }
//                     stopExecution = true;
//                 }
//                 continue;
//             }
//         }
//     }
//     ofs.close();

//     int count_iteration_reading = 0;

// #pragma omp parallel for
//     for (int time_point = 0; time_point < time.size(); time_point++)
//     {
//         if (stopExecution)
//         {
//             continue;
//         }
//         ifstream ifs("integrate_MySystem.bin", ios::binary | ios::in);
//         try
//         {
//             double weight_x;
//             double weight_y;
//             double weight;
//             double sum1 = 0;
//             double sum2 = 0;
//             for (int i = 0; i <= Nx; i++)
//             {
//                 for (int j = 0; j <= Ny; j++)
//                 {
//                     const double px = lower_limitX + (h_x * i);
//                     const double py = lower_limitY + (h_y * j);

//                     Matrix2cd Rho = retrieveI(i, j, objects[i][j], ifs)[time_point];
//                     double func_value1 = ((func1(time[time_point], px, py) * Rho).trace().real());
//                     double func_value2 = ((func2(time[time_point], px, py) * Rho).trace().real());

//                     if (isnan(func_value1) || isnan(func_value2))
//                     {
//                         throw runtime_error("NAN value appeared at in integration at \n"
//                                             "time = " +
//                                             to_string(time[time_point]) + "\n"
//                                                                           "px = " +
//                                             to_string(px) + "\n"
//                                                             "py = " +
//                                             to_string(py));
//                     }

//                     if (i == 0 || i == Nx)
//                     {
//                         weight_x = 1;
//                     }
//                     else if (i % 3 == 0)
//                     {
//                         weight_x = 2;
//                     }
//                     else
//                     {
//                         weight_x = 3;
//                     }
//                     if (j == 0 || j == Ny)
//                     {
//                         weight_y = 1;
//                     }
//                     else if (j % 3 == 0)
//                     {
//                         weight_y = 2;
//                     }
//                     else
//                     {
//                         weight_y = 3;
//                     }
//                     weight = (9.0 / 64.0) * h_x * h_y * weight_x * weight_y * sin(M_PI / 2.0);
//                     sum1 += weight * func_value1;
//                     sum2 += weight * func_value2;
//                 }
//             }
//             integration_vals1[time_point] = sum1;
//             integration_vals2[time_point] = sum2;
//         }
//         catch (...)
//         {
//             ifs.close();
// #pragma omp critical
//             {
//                 if (!excPtr) // Only store the first exception we catch
//                 {
//                     excPtr = current_exception();
//                 }
//                 stopExecution = true;
//             }
//             continue;
//         }
//         ifs.close();
//     }
//     if (excPtr)
//     {
//         rethrow_exception(excPtr);
//     }
// }

void errorHandler(const char *reason, const char *file, int line, int gsl_errno)
{
    throw runtime_error("Reason: " + string(reason) + " , File: " + string(file) + " ,Line: " + to_string(line) + " ,Error no.: " + to_string(gsl_errno));
}

struct Params
{
    double t, target_value;
};

double epsilon_px(double px, void *params)
{
    Params *eq_params = static_cast<Params *>(params);
    double py = eq_params->target_value;
    double t = eq_params->t;
    return (epsilon(t, px, py) - ef);
}
double epsilon_py(double py, void *params)
{
    Params *eq_params = static_cast<Params *>(params);
    double px = eq_params->target_value;
    double t = eq_params->t;
    return (epsilon(t, px, py) - ef);
}
double epsilonG_px(double px, void *params)
{
    Params *eq_params = static_cast<Params *>(params);
    double py = eq_params->target_value;
    double t = eq_params->t;
    return (epsilon_G(t, px, py) - ef);
}
double epsilonG_py(double py, void *params)
{
    Params *eq_params = static_cast<Params *>(params);
    double px = eq_params->target_value;
    double t = eq_params->t;
    return (epsilon_G(t, px, py) - ef);
}

vector<double> P_x;
vector<double> P_y;
void coordinates(const double t, const double pxl, const double pxh, const double pyl, const double pyh)
{
    if ((pxl >= pxh) || (pyl >= pyh))
    {
        throw domain_error("in coordinates check lower and upper valueas of x an y");
    }

    gsl_set_error_handler(&errorHandler);
    const int N = 100;
    vector<double> P_X;
    vector<double> P_Y;

    int Count_xx = (int)(N * abs(pxh - pxl)) > N ? (int)(N * abs(pxh - pxl)) : N;
    double step_xx = abs(pxh - pxl) / Count_xx;
    int Count_yy = (int)(N * abs(pyh - pyl)) > N ? (int)(N * abs(pyh - pyl)) : N;
    double step_yy = abs(pyh - pyl) / Count_yy;

    for (int i = 0; i <= Count_xx; i++)
    {
        double px = pxl + (step_xx * i);
        int Count_y = (int)(N * abs(pyh - pyl)) > N ? (int)(N * abs(pyh - pyl)) : N;
        double step_y = abs(pyh - pyl) / Count_y;
        int j_l = 0;
        for (int j = 0; j < Count_y; j++)
        {
            double py_l = pyl + (step_y * j_l);
            double py_h = pyl + (step_y * (j + 1));

            if ((epsilon_G(t, px, py_l) - ef) * (epsilon_G(t, px, py_h) - ef) < 0)
            {
                int status;
                int iter = 0, max_iter = 100;
                const gsl_root_fsolver_type *T;
                gsl_root_fsolver *s;
                double r = 0;
                gsl_function F;
                Params params = {t, px};
                F.function = &epsilonG_py;
                F.params = &params;

                T = gsl_root_fsolver_brent;
                s = gsl_root_fsolver_alloc(T);
                try
                {
                    gsl_root_fsolver_set(s, &F, py_l, py_h);

                    do
                    {
                        iter++;
                        status = gsl_root_fsolver_iterate(s);
                        r = gsl_root_fsolver_root(s);
                        py_l = gsl_root_fsolver_x_lower(s);
                        py_h = gsl_root_fsolver_x_upper(s);
                        status = gsl_root_test_interval(py_l, py_h, eps_abs, eps_rel);

                        if (status == GSL_SUCCESS)
                        {
                            P_X.push_back(px);
                            P_Y.push_back(r);
                        }
                    } while (status == GSL_CONTINUE && iter < max_iter);

                    j_l = j + 1;
                }
                catch (const runtime_error &e)
                {
                    gsl_root_fsolver_free(s);
                    throw runtime_error("Root finding error. \n"
                                        "px = " +
                                        to_string(px) + "\n"
                                                        "py_l = " +
                                        to_string(py_l) + "\n"
                                                          "py_h = " +
                                        to_string(py_h) + "\n" +
                                        string(e.what()));
                }
                if (status != GSL_SUCCESS)
                {
                    throw runtime_error("Root finding not converged in " + to_string(max_iter) + " number of iteration. \n"
                                                                                                 "px = " +
                                        to_string(px) + "\n"
                                                        "py_l = " +
                                        to_string(py_l) + "\n"
                                                          "py_h = " +
                                        to_string(py_h) + "\n"
                                                          "root : " +
                                        to_string(r));
                }
            }
            else if (abs(epsilon_G(t, px, py_l) - ef) < 1e-10)
            {
                double r = py_l;
                P_X.push_back(px);
                P_Y.push_back(r);
                j_l = j + 1;
            }
            else if (abs(epsilon_G(t, px, py_h) - ef) < 1e-10)
            {
                double r = py_h;
                P_X.push_back(px);
                P_Y.push_back(r);
                j_l = j + 1;
            }

            if ((epsilon(t, px, py_l) - ef) * (epsilon(t, px, py_h) - ef) < 0)
            {
                int status;
                int iter = 0, max_iter = 100;
                const gsl_root_fsolver_type *T;
                gsl_root_fsolver *s;
                double r = 0;
                gsl_function F;
                Params params = {t, px};
                F.function = &epsilon_py;
                F.params = &params;

                T = gsl_root_fsolver_brent;
                s = gsl_root_fsolver_alloc(T);
                try
                {
                    gsl_root_fsolver_set(s, &F, py_l, py_h);

                    do
                    {
                        iter++;
                        status = gsl_root_fsolver_iterate(s);
                        r = gsl_root_fsolver_root(s);
                        py_l = gsl_root_fsolver_x_lower(s);
                        py_h = gsl_root_fsolver_x_upper(s);
                        status = gsl_root_test_interval(py_l, py_h, eps_abs, eps_rel);

                        if (status == GSL_SUCCESS)
                        {
                            P_X.push_back(px);
                            P_Y.push_back(r);
                        }
                    } while (status == GSL_CONTINUE && iter < max_iter);

                    j_l = j + 1;
                }
                catch (const runtime_error &e)
                {
                    gsl_root_fsolver_free(s);
                    throw runtime_error("Root finding error. \n"
                                        "px = " +
                                        to_string(px) + "\n"
                                                        "py_l = " +
                                        to_string(py_l) + "\n"
                                                          "py_h = " +
                                        to_string(py_h) + "\n" +
                                        string(e.what()));
                }
                if (status != GSL_SUCCESS)
                {
                    throw runtime_error("Root finding not converged in " + to_string(max_iter) + " number of iteration. \n"
                                                                                                 "px = " +
                                        to_string(px) + "\n"
                                                        "py_l = " +
                                        to_string(py_l) + "\n"
                                                          "py_h = " +
                                        to_string(py_h) + "\n"
                                                          "root : " +
                                        to_string(r));
                }
            }
            else if (abs(epsilon(t, px, py_l) - ef) < 1e-10)
            {
                double r = py_l;
                P_X.push_back(px);
                P_Y.push_back(r);
                j_l = j + 1;
            }
            else if (abs(epsilon(t, px, py_h) - ef) < 1e-10)
            {
                double r = py_h;
                P_X.push_back(px);
                P_Y.push_back(r);
                j_l = j + 1;
            }
        }
    }
    for (int i = 0; i <= Count_yy; i++)
    {
        double py = pyl + (step_yy * i);
        int Count_x = (int)(N * abs(pxh - pxl)) > N ? (int)(N * abs(pxh - pxl)) : N;
        double step_x = abs(pxh - pxl) / Count_x;
        int j_l = 0;
        for (int j = 0; j < Count_x; j++)
        {
            double px_l = pxl + (step_x * j_l);
            double px_h = pxl + (step_x * (j + 1));
            if ((epsilon_G(t, px_l, py) - ef) * (epsilon_G(t, px_h, py) - ef) < 0)
            {
                int status;
                int iter = 0, max_iter = 100;
                const gsl_root_fsolver_type *T;
                gsl_root_fsolver *s;
                double r = 0;
                gsl_function F;
                Params params = {t, py};
                F.function = &epsilonG_px;
                F.params = &params;

                T = gsl_root_fsolver_brent;
                s = gsl_root_fsolver_alloc(T);
                try
                {
                    gsl_root_fsolver_set(s, &F, px_l, px_h);

                    do
                    {
                        iter++;
                        status = gsl_root_fsolver_iterate(s);
                        r = gsl_root_fsolver_root(s);
                        px_l = gsl_root_fsolver_x_lower(s);
                        px_h = gsl_root_fsolver_x_upper(s);
                        status = gsl_root_test_interval(px_l, px_h, eps_abs, eps_rel);

                        if (status == GSL_SUCCESS)
                        {
                            P_X.push_back(r);
                            P_Y.push_back(py);
                        }
                    } while (status == GSL_CONTINUE && iter < max_iter);

                    j_l = j + 1;
                }

                catch (const runtime_error &e)
                {
                    gsl_root_fsolver_free(s);
                    throw runtime_error("Root finding error. \n"
                                        "py = " +
                                        to_string(py) + "\n"
                                                        "px_l = " +
                                        to_string(px_l) + "\n"
                                                          "px_h = " +
                                        to_string(px_h) + "\n" +
                                        string(e.what()));
                }
                if (status != GSL_SUCCESS)
                {
                    throw runtime_error("Root finding not converged in " + to_string(max_iter) + " number of iteration. \n"
                                                                                                 "px = " +
                                        to_string(py) + "\n"
                                                        "px_l = " +
                                        to_string(px_l) + "\n"
                                                          "px_h = " +
                                        to_string(px_h) + "\n"
                                                          "root : " +
                                        to_string(r));
                }
            }
            else if (abs(epsilon_G(t, px_l, py) - ef) < 1e-10)
            {
                double r = px_l;
                P_X.push_back(r);
                P_Y.push_back(py);
                j_l = j + 1;
            }
            else if (abs(epsilon_G(t, px_h, py) - ef) < 1e-10)
            {
                double r = px_h;
                P_X.push_back(r);
                P_Y.push_back(py);
                j_l = j + 1;
            }

            if ((epsilon(t, px_l, py) - ef) * (epsilon(t, px_h, py) - ef) < 0)
            {
                int status;
                int iter = 0, max_iter = 100;
                const gsl_root_fsolver_type *T;
                gsl_root_fsolver *s;
                double r = 0;
                gsl_function F;
                Params params = {t, py};
                F.function = &epsilon_px;
                F.params = &params;

                T = gsl_root_fsolver_brent;
                s = gsl_root_fsolver_alloc(T);
                try
                {
                    gsl_root_fsolver_set(s, &F, px_l, px_h);

                    do
                    {
                        iter++;
                        status = gsl_root_fsolver_iterate(s);
                        r = gsl_root_fsolver_root(s);
                        px_l = gsl_root_fsolver_x_lower(s);
                        px_h = gsl_root_fsolver_x_upper(s);
                        status = gsl_root_test_interval(px_l, px_h, eps_abs, eps_rel);

                        if (status == GSL_SUCCESS)
                        {
                            P_X.push_back(r);
                            P_Y.push_back(py);
                        }
                    } while (status == GSL_CONTINUE && iter < max_iter);

                    j_l = j + 1;
                }

                catch (const runtime_error &e)
                {
                    gsl_root_fsolver_free(s);
                    throw runtime_error("Root finding error. \n"
                                        "py = " +
                                        to_string(py) + "\n"
                                                        "px_l = " +
                                        to_string(px_l) + "\n"
                                                          "px_h = " +
                                        to_string(px_h) + "\n" +
                                        string(e.what()));
                }
                if (status != GSL_SUCCESS)
                {
                    throw runtime_error("Root finding not converged in " + to_string(max_iter) + " number of iteration. \n"
                                                                                                 "px = " +
                                        to_string(py) + "\n"
                                                        "px_l = " +
                                        to_string(px_l) + "\n"
                                                          "px_h = " +
                                        to_string(px_h) + "\n"
                                                          "root : " +
                                        to_string(r));
                }
            }
            else if (abs(epsilon(t, px_l, py) - ef) < 1e-10)
            {
                double r = px_l;
                P_X.push_back(r);
                P_Y.push_back(py);
                j_l = j + 1;
            }
            else if (abs(epsilon(t, px_h, py) - ef) < 1e-10)
            {
                double r = px_h;
                P_X.push_back(r);
                P_Y.push_back(py);
                j_l = j + 1;
            }
        }
    }

    P_x.clear();
    P_y.clear();
    for (int i = 0; i < P_X.size(); i++)
    {
        bool found = false;
        for (int j = i + 1; j < P_X.size(); j++)
        {
            if ((abs(P_X[i] - P_X[j]) < 1e-10) && (abs(P_Y[i] - P_Y[j]) < 1e-10))
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            P_x.push_back(P_X[i]);
            P_y.push_back(P_Y[i]);
        }
    }
}

int main()
{
    initialization();
    const int TIME_DIVISION = 100;
    const int division = (int)(TIME_DIVISION * (final_time - initial_time)) > TIME_DIVISION ? (int)(TIME_DIVISION * (final_time - initial_time)) : TIME_DIVISION;
    const double h1 = (abs(final_time - initial_time)) / division;

    vector<double> Time(division + 1);

    for (int i = 0; i <= division; i++)
    {
        Time[i] = initial_time + (i * h1);
    }
    ti = *min_element(Time.begin(), Time.end());
    tf = *max_element(Time.begin(), Time.end());

    cout << "calculations started ****************" << endl;

    printCurrentDateTime();
    const auto start = chrono::high_resolution_clock::now();

    ofstream file;

    try
    {
        cout << "started coordinates" << endl;
        coordinates(ti, -2, 2, -2, 2);
        auto end = chrono::high_resolution_clock::now();
        double elapsed_time = chrono::duration<double>(end - start).count();
        cout << "Elapsed time in coordinates: " << elapsed_time << " seconds." << endl;
    }
    catch (const exception &e)
    {
        cerr << setprecision(precision) << "Exception: " + string(e.what()) << endl;
        auto end = chrono::high_resolution_clock::now();
        double elapsed_time = chrono::duration<double>(end - start).count();
        cout << "Elapsed time: " << elapsed_time << " seconds." << endl;
        return 1;
    }
    catch (...)
    {
        cerr << "unknown error occured" << endl;
        auto end = chrono::high_resolution_clock::now();
        double elapsed_time = chrono::duration<double>(end - start).count();
        cout << "Elapsed time: " << elapsed_time << " seconds." << endl;
        return 1;
    }

    file.open("results/px_py.txt");

    if (file.is_open())
    {
        file << setprecision(precision);

        file << left << setw(25) << "px";
        file << left << setw(25) << "py";
        file << endl;
    }

    for (int i = 0; i < P_x.size(); i++)
    {
        double result_x, result_y;

        result_x = P_x[i];
        result_y = P_y[i];
        if (file.is_open())
        {
            file << left << setw(25) << result_x;
            file << left << setw(25) << result_y;
            file << endl;
        }
    }

    file.close();

    cout << "DensityMatrix surface integration started" << endl;
    cout << "************************************************************************************************" << endl;
    const int integral_divisionX = Nx / 300;
    const int integral_divisionY = Ny / 300;

    cout << "integral_divisionX = " << integral_divisionX << endl;
    cout << "integral_divisionY = " << integral_divisionY << endl;

    const double DX = abs(upper_limitX - lower_limitX) / integral_divisionX;
    const double DY = abs(upper_limitY - lower_limitY) / integral_divisionY;

    vector<double> JX(Time.size(), 0.0);
    vector<double> JY(Time.size(), 0.0);
    vector<double> SX(Time.size(), 0.0);
    vector<double> SY(Time.size(), 0.0);
    for (int i_x = 0; i_x < integral_divisionX; i_x++)
    {
        for (int j_y = 0; j_y < integral_divisionY; j_y++)
        {
            double llX = lower_limitX + (DX * i_x);
            double llY = lower_limitY + (DY * j_y);
            double ulX = llX + DX;
            double ulY = llY + DY;
            vector<double> JX_temp(Time.size());
            vector<double> JY_temp(Time.size());
            vector<double> SX_temp(Time.size());
            vector<double> SY_temp(Time.size());
            try
            {
                cout << "started J*********************" << endl;
                integration(llX, ulX, llY, ulY, Time, jx, jy, JX_temp, JY_temp);
                auto end = chrono::high_resolution_clock::now();
                double elapsed_time = chrono::duration<double>(end - start).count();
                cout << setprecision(precision) << "Elapsed time of J calculation : " << elapsed_time << " seconds" << endl;

                cout << "started S*********************" << endl;
                integration(llX, ulX, llY, ulY, Time, Sx, Sy, SX_temp, SY_temp);
                cout << setprecision(precision) << "Elapsed time of S calculation : " << elapsed_time << " seconds" << endl;
            }
            catch (const exception &e)
            {
                cerr << setprecision(precision) << "Exception : " + string(e.what()) << endl;
                auto end = chrono::high_resolution_clock::now();
                double elapsed_time = chrono::duration<double>(end - start).count();
                cout << setprecision(precision) << "Elapsed time : " << elapsed_time << " seconds." << endl;
                return 1;
            }
            transform(JX_temp.begin(), JX_temp.end(), JX.begin(), JX.begin(), plus<double>());
            transform(JY_temp.begin(), JY_temp.end(), JY.begin(), JY.begin(), plus<double>());
            transform(SX_temp.begin(), SX_temp.end(), SX.begin(), SX.begin(), plus<double>());
            transform(SY_temp.begin(), SY_temp.end(), SY.begin(), SY.begin(), plus<double>());
        }
    }

    file.open("results/datas.csv");
    try
    {
        if (file.is_open())
        {
            file << setprecision(precision);

            file << "Time"
                 << ","
                 << "Jx"
                 << ","
                 << "Jy"
                 << ","
                 << "Sx"
                 << ","
                 << "Sy"
                 << ","
                 << "Ax"
                 << ","
                 << "Ay" << endl;
            for (int time_point = 0; time_point < JX.size(); time_point++)
            {
                file << Time[time_point]
                     << ","
                     << JX[time_point]
                     << ","
                     << JY[time_point]
                     << ","
                     << SX[time_point]
                     << ","
                     << SY[time_point]
                     << ","
                     << Ax(Time[time_point])
                     << ","
                     << Ay(Time[time_point]) << endl;
            }
        }
        else
        {
            throw runtime_error("Error: could not open file for parameters");
        }
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    file.close();

    file.open("results/parameters.txt");
    try
    {
        if (file.is_open())
        {
            file << setprecision(precision);
            // **************************************

            bool flatnessX, flatnessY;
            string delayX, delayY;

            if (((A0E_x == 0 && A0_x == 0) && pulse_wrapper_x == "add") || (A0E_x == 0 && pulse_wrapper_x == "envelope"))
            {
                flatnessX = true;
            }
            else
            {
                flatnessX = false;
            }

            if (((A0E_y == 0 && A0_y == 0) && pulse_wrapper_y == "add") || (A0E_y == 0 && pulse_wrapper_y == "envelope"))
            {
                flatnessY = true;
            }
            else
            {
                flatnessY = false;
            }

            if (A0_x == 0 && A0_y == 0)
            {
                delayX = "delayX = " + to_string(x_delay);
                delayY = "delayY = " + to_string(y_delay);
            }
            else if (A0_x != 0 && A0_y != 0)
            {
                delayX = "phiX = " + to_string(phi_x);
                delayY = "phiY = " + to_string(phi_y);
            }
            else if (A0_x == 0 && A0_y != 0)
            {
                delayX = "phiXy = " + to_string(phi_y);
                delayY = "phiY = " + to_string(phi_y);
            }
            else if (A0_x != 0 && A0_y == 0)
            {
                delayX = "phiX = " + to_string(phi_x);
                delayY = "phiYx = " + to_string(phi_x);
            }

            // string directory = "hbar=" + to_string(hbar) + ",t1=" + to_string(t1) + ",t2=" + to_string(t2) + ",M=" + to_string(M) + ",a=" + to_string(a) + ",q=" + to_string(q) + ",delX=" + to_string(delta_x);
            file << "hbar = " << hbar << endl;
            file << "t1 = " << t1 << endl;
            file << "t2 = " << t2 << endl;
            file << "M = " << M << endl;
            file << "a = " << a << endl;
            file << "q = " << q << endl;
            file << "delX = " << delta_x << endl;

            // string x_component_dir = ",A0XE=" + to_string(A0E_x);
            file << "A0XE = " << A0E_x << endl;
            if (!flatnessX)
            {
                // x_component_dir += ",pulseX=" + pulse_type_x + "_" + pulse_wrapper_x;
                file << "pulseX = " << pulse_type_x + "_" + pulse_wrapper_x << endl;

                if (pulse_wrapper_x == "envelope")
                {
                    if (pulse_type_x == "gaussian")
                    {
                        file << "sigmaX = " << sigma_x << endl;
                        file << "ntt_sigmaX = " << ntt_sigmaX << endl;
                        if (!flatnessY)
                        {
                            // x_component_dir += ",nuX=" + to_string(nu_x) + "," + delayX;
                            file << "nuX = " << nu_x << endl;
                            file << delayX << endl;
                        }
                        else
                        {
                            // x_component_dir += ",nuX=" + to_string(nu_x);
                            file << "nuX = " << nu_x << endl;
                        }
                    }
                    else if (pulse_type_x == "sin^2")
                    {
                        if (!flatnessY)
                        {
                            // x_component_dir += ",nuX=" + to_string(nu_x) + ",NOCX=" + to_string(number_of_cycle_x) + "," + delayX;
                            file << "nuX = " << nu_x << endl;
                            file << "NOCX = " << number_of_cycle_x << endl;
                            file << delayX << endl;
                        }
                        else
                        {
                            // x_component_dir += ",nuX=" + to_string(nu_x) + ",NOCX=" + to_string(number_of_cycle_x);
                            file << "nuX = " << nu_x << endl;
                            file << "NOCX = " << number_of_cycle_x << endl;
                        }
                    }
                    else if (pulse_type_x == "box_type")
                    {
                        if (!flatnessY)
                        {
                            // x_component_dir += ",nuX=" + to_string(nu_x) + ",NOCX=" + to_string(number_of_cycle_x) + "," + delayX + ",nuXE=" + to_string(nu_e_x);
                            file << "nuX = " << nu_x << endl;
                            file << "NOCX = " << number_of_cycle_x << endl;
                            file << delayX << endl;
                            file << "nuXE = " << nu_e_x << endl;
                        }
                        else
                        {
                            // x_component_dir += ",nuX=" + to_string(nu_x) + ",NOCX=" + to_string(number_of_cycle_x) + ",nuXE=" + to_string(nu_e_x);
                            file << "nuX = " << nu_x << endl;
                            file << "NOCX = " << number_of_cycle_x << endl;
                            file << "nuXE = " << nu_e_x << endl;
                        }
                    }
                    else
                    {
                        throw runtime_error("Invalid pulse_type");
                    }
                }
                else if (pulse_wrapper_x == "add")
                {
                    // x_component_dir += ",A0X=" + to_string(A0_x);
                    file << "A0X = " << A0_x << endl;
                    if (A0_x != 0)
                    {
                        file << "nuX = " << nu_x << endl;
                        file << "NOCX = " << number_of_cycle_x << endl;
                    }
                    if (pulse_type_x == "gaussian")
                    {
                        file << "sigmaX = " << sigma_x << endl;
                        file << "ntt_sigmaX = " << ntt_sigmaX << endl;
                        if (!flatnessY)
                        {
                            file << delayX << endl;
                        }
                    }
                    else if (pulse_type_x == "sin^2")
                    {
                        if (!flatnessY)
                        {
                            file << delayX << endl;
                        }
                    }
                    else if (pulse_type_x == "box_type")
                    {
                        if (!flatnessY)
                        {
                            file << delayX << endl;
                            file << "nuXE = " << nu_e_x << endl;
                            file << "ETPX = " << ETP_x << endl;
                        }
                        else
                        {
                            file << "nuXE = " << nu_e_x << endl;
                            file << "ETPX = " << ETP_x << endl;
                        }
                    }
                    else
                    {
                        throw runtime_error("Invalid pulse_type");
                    }
                }
            }
            else
            {
                // x_component_dir += ",pulseX=flat";
                file << "pulseX = flat";
            }

            // string y_component_dir = ",A0YE=" + to_string(A0E_y);
            file << "A0YE = " << A0E_y << endl;
            if (!flatnessY)
            {
                // y_component_dir += ",pulseY=" + pulse_type_y + "_" + pulse_wrapper_y;
                file << "pulseY = " << pulse_type_y + "_" + pulse_wrapper_y << endl;
                if (pulse_wrapper_y == "envelope")
                {
                    if (pulse_type_y == "gaussian")
                    {
                        file << "sigmaY = " << sigma_y << endl;
                        file << "ntt_sigmaY = " << ntt_sigmaY << endl;
                        if (!flatnessX)
                        {
                            // y_component_dir += ",nuY=" + to_string(nu_y) + "," + delayY;
                            file << "nuY = " << nu_y << endl;
                            file << delayY << endl;
                        }
                        else
                        {
                            // y_component_dir += ",nuY=" + to_string(nu_y);
                            file << "nuY = " << nu_y << endl;
                        }
                    }
                    else if (pulse_type_y == "sin^2")
                    {
                        if (!flatnessX)
                        {
                            // y_component_dir += ",nuY=" + to_string(nu_y) + ",NOCY=" + to_string(number_of_cycle_y) + "," + delayY;
                            file << "nuY = " << nu_y << endl;
                            file << "NOCY = " << number_of_cycle_y << endl;
                            file << delayY << endl;
                        }
                        else
                        {
                            // y_component_dir += ",nuY=" + to_string(nu_y) + ",NOCY=" + to_string(number_of_cycle_y);
                            file << "nuY = " << nu_y << endl;
                            file << "NOCY = " << number_of_cycle_y << endl;
                        }
                    }
                    else if (pulse_type_y == "box_type")
                    {
                        if (!flatnessX)
                        {
                            // y_component_dir += ",nuY=" + to_string(nu_y) + ",NOCY=" + to_string(number_of_cycle_y) + "," + delayY + ",nuYE=" + to_string(nu_e_y);
                            file << "nuY = " << nu_y << endl;
                            file << "NOCY = " << number_of_cycle_y << endl;
                            file << delayY << endl;
                            file << "nuYE =" << nu_e_y << endl;
                        }
                        else
                        {
                            // y_component_dir += ",nuY=" + to_string(nu_y) + ",NOCY=" + to_string(number_of_cycle_y) + ",nuYE=" + to_string(nu_e_y);
                            file << "nuY = " << nu_y << endl;
                            file << "NOCY = " << number_of_cycle_y << endl;
                            file << "nuYE =" << nu_e_y << endl;
                        }
                    }
                    else
                    {
                        throw runtime_error("Invalid pulse_type");
                    }
                }
                else if (pulse_wrapper_y == "add")
                {
                    file << "A0Y = " << A0_y << endl;
                    if (A0_y != 0)
                    {
                        file << "nuY = " << nu_y << endl;
                        file << "NOCY = " << number_of_cycle_y << endl;
                    }
                    if (pulse_type_y == "gaussian")
                    {
                        file << "sigmaY = " << sigma_y << endl;
                        file << "ntt_sigmaY = " << ntt_sigmaY << endl;
                        if (!flatnessX)
                        {
                            file << delayY << endl;
                        }
                    }
                    else if (pulse_type_y == "sin^2")
                    {
                        if (!flatnessX)
                        {
                            file << delayY << endl;
                        }
                    }
                    else if (pulse_type_y == "box_type")
                    {
                        if (!flatnessX)
                        {
                            file << delayY << endl;
                            file << "nuYE = " << nu_e_y << endl;
                            file << "ETPY = " << ETP_y << endl;
                        }
                        else
                        {
                            file << "nuYE = " << nu_e_y << endl;
                            file << "ETPY = " << ETP_y << endl;
                        }
                    }
                    else
                    {
                        throw runtime_error("Invalid pulse_type");
                    }
                }
            }
            else
            {
                // y_component_dir += ",pulseY=flat";
                file << "pulseY = flat" << endl;
            }

            // directory + y_component_dir += ",nen=" + to_string(y_co) + ",=" += " + to_string(ng(dephase) + ", Nx = " + to_string(Nx) + ", Ny = " + to_string(Ny) + ", llX = " + to_string(l) + ", ETPY = " + to_string(ETP_y)itX) + ", ulX = " + to_string(upper_limitX) + ", llY = " + to_string(lower_limitY) + ", ulY = " + to_string(upper_limitY) + ", ETPO = " + to_string(extra_time_percentage);

            file << "ef = " << ef << endl;
            file << "dephase = " << dephase << endl;
            file << "Nx = " << Nx << endl;
            file << "Ny = " << Ny << endl;
            file << "lower_limitX = " << lower_limitX << endl;
            file << "upper_limitX = " << upper_limitX << endl;
            file << "lower_limitY = " << lower_limitY << endl;
            file << "upper_limitY = " << upper_limitY << endl;
            file << "extra_time_percentage = " << extra_time_percentage << endl;
            file << "Total surface area at t = " << ti << " is : " << SurfaceIntegration(ti) << endl;
            // *********************************
        }
        else
        {
            throw runtime_error("could not open file for parameters");
        }
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    file.close();

    // **************************************************************************************************************

    // cout << "surface area = " <<SurfaceIntegration(ti)<< endl;

    // **************************************************************************************************************
    cout << "calculations ended ****************" << endl;

    printCurrentDateTime();
    auto end = chrono::high_resolution_clock::now();

    double elapsed_time = chrono::duration<double>(end - start).count();

    cout << "Elapsed time: " << elapsed_time << " seconds." << endl;

    return 0;
}
