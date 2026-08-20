#include <omp.h>
#include <fstream>
#include "boost/numeric/odeint.hpp"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_roots.h"
#include "./pulse_type.h"

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
    std::cout << "date: " << put_time(timeStruct, "%d-%m-%Y") << ", time: " << put_time(timeStruct, "%H-%M-%S") << endl;
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
const double alpha = 1;
const double gama = 1;
const double w = 0.1;
const double a = 1;
const double q = 1;

const double ef = 1.00498756;
const double dephase = 0.0;

// NX and NY should be multiple of Nx and Ny in this case. This can be modified in the integration function if
// needed of memory management
const int NX = 1800;
const int NY = 1800;
const int Nx = 300; // appears in integration function
const int Ny = 300; // appears in integration function
const double lower_limitX = -2;
const double upper_limitX = 2;
const double lower_limitY = -2;
const double upper_limitY = 2;

const double A0X = 0.5;                   // Amplitude of pulse
const double nuX = 0.2;                   // Frequency of pulse
const string pulse_typeX = "sin^2";       // Type of pulse
const string pulse_wrapperX = "envelope"; // "NA" if not and envelope. "envelope" if an envelope of sin pulse
const double sigmaX = 0.1;                // in case of gaussian pulse is the SD
const double ntt_sigmaX = 1.0;            // in case of gaussian pulse, total time = ntt_sigma * sigma
const double ETPX = 100.0;                // Extra time percentage
const double delayX = 0.0;                // consider it as initial time t_i, i.e. from 0 how long it delayed to start
const int number_of_cycleX = 5;           // number of cycles of the pulse
const double durationX = 5;               // duration of the pulse for linear pulse

const double A0Y = 0.0;
const double nuY = 1.0;
const string pulse_typeY = "sin^2";
const string pulse_wrapperY = "envelope";
const double sigmaY = 0.1;
const double ntt_sigmaY = 1.0;
const double ETPY = 0.0;
const double delayY = 0.0;
const int number_of_cycleY = 1;
const double durationY = 5; // duration of the pulse for linear pulse

PulseType PulseObjectX(pulse_typeX, pulse_wrapperX, A0X, sigmaX, ntt_sigmaX, nuX, ETPX, delayX, number_of_cycleX, durationX);
PulseType PulseObjectY(pulse_typeY, pulse_wrapperY, A0Y, sigmaY, ntt_sigmaY, nuY, ETPY, delayY, number_of_cycleY, durationY);
D2Pulse Pulse(PulseObjectX, PulseObjectY);

const double ti = min(PulseObjectX.ti, PulseObjectY.ti);
const double tf = max(PulseObjectX.tf, PulseObjectY.tf);

double Ax(const double t)
{
    return (Pulse.A(t))(0);
}
double Ay(const double t)
{
    return (Pulse.A(t))(1);
}

double sigmax_term(const double t, const double px, const double py)
{
    (void)t;
    (void)px;
    (void)py;
    double sigmax_term = (gama + gama) / 2.0;
    return sigmax_term;
}
double sigmay_term(const double t, const double px, const double py)
{
    (void)t;
    (void)px;
    (void)py;
    double sigmay_term = 0.0;
    return sigmay_term;
}
double sigmaz_term(const double t, const double px, const double py)
{
    (void)t;
    double sigmaz_term = ((alpha * (pow(px - q * Ax(t), 2) + pow((py - q * Ay(t)), 2))) - w) / 2.0;
    return (sigmaz_term);
}
double identity_term(const double t, const double px, const double py)
{
    (void)t;
    double identity_term = ((alpha * (pow(px - q * Ax(t), 2) + pow((py - q * Ay(t)), 2))) + w) / 2.0;
    return identity_term;
}

Matrix2cd H(const double t, const double px, const double py)
{
    Matrix2cd Hamiltonian;
    Hamiltonian = (sigmax_term(t, px, py) * PauliMatrix_1) + (sigmay_term(t, px, py) * PauliMatrix_2) + (sigmaz_term(t, px, py) * PauliMatrix_3) + (identity_term(t, px, py) * Identity_2);
    return Hamiltonian;
}

pair<Matrix2cd, Vector2d> EigenSystem(const double t, const double px, const double py)
{
    SelfAdjointEigenSolver<Matrix2cd> es(H(t, px, py));
    Vector2d vals = es.eigenvalues().real();
    vector<int> idx(vals.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&vals](int i, int j)
         { return vals[i] < vals[j]; });
    Vector2d values;
    Matrix2cd vectors;
    for (int i = 0; i < vals.size(); i++)
    {
        values[i] = vals.coeff(idx[i]);
        vectors.col(i) = es.eigenvectors().col(idx[i]);
    }
    return {vectors, values};
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
    auto [EVecs, EVals] = EigenSystem(t, px, py);
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
        UM = EVecs;
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
    (void)t;
    (void)px;
    (void)py;
    double Dxsigma_x = 0.0;
    return Dxsigma_x;
}
double Dxsigma_y(const double t, const double px, const double py)
{
    (void)t;
    (void)px;
    (void)py;
    double Dxsigma_y = 0.0;
    return Dxsigma_y;
}
double Dxsigma_z(const double t, const double px, const double py)
{
    (void)t;
    (void)py;
    double Dxsigma_z = alpha * (px - q * Ax(t));
    return (Dxsigma_z);
}
double Dxidentity(const double t, const double px, const double py)
{
    (void)t;
    (void)py;
    double Dxidentity = alpha * (px - q * Ax(t));
    return Dxidentity;
}

double Dysigma_x(const double t, const double px, const double py)
{
    (void)t;
    (void)px;
    (void)py;
    double Dysigma_x = 0.0;
    return Dysigma_x;
}
double Dysigma_y(const double t, const double px, const double py)
{
    (void)t;
    (void)px;
    (void)py;
    double Dysigma_y = 0.0;
    return Dysigma_y;
}
double Dysigma_z(const double t, const double px, const double py)
{
    (void)t;
    (void)px;
    double Dysigma_z = alpha * (py - q * Ay(t));
    return (Dysigma_z);
}
double Dyidentity(const double t, const double px, const double py)
{
    (void)t;
    (void)px;
    double Dyidentity = alpha * (py - q * Ay(t));
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
    Matrix2cd ireb{{fermi_function(epsilon_G(t, px, py) - ef), 0.0},
                   {0.0, fermi_function(epsilon(t, px, py) - ef)}};
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
        (void)t;
        state_type psudo_psi = y;
        Matrix2cd RHO{{psudo_psi[0], psudo_psi[1]},
                      {psudo_psi[2], psudo_psi[3]}};

        I.push_back(RHO);
    }
};

class integrate_MySystem
{
private:
    vector<double> time_range;
    double px, py;
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
};

void integration(const double lower_limitX, const double upper_limitX, const double lower_limitY, const double upper_limitY, const vector<double> time, Matrix2cd (*func1)(double, double, double), Matrix2cd (*func2)(double, double, double), vector<double> &integration_vals1, vector<double> &integration_vals2)
{
    atomic<bool> stopExecution(false);
    exception_ptr excPtr = nullptr;
    if ((lower_limitX >= upper_limitX) || (lower_limitY >= upper_limitY))
    {
        throw domain_error("in integration check lower and upper limit of x and y carefully");
    }

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
    for (size_t time_point = 0; time_point < time.size(); time_point++)
    {
        if (stopExecution)
        {
            continue;
        }
        try
        {
            double sum1 = 0.0;
            double sum2 = 0.0;
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

                    sum1 += func_value1;
                    sum2 += func_value2;
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
    for (size_t i = 0; i < P_X.size(); i++)
    {
        bool found = false;
        for (size_t j = i + 1; j < P_X.size(); j++)
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
    const int nthreads = omp_get_max_threads();
    printf("Using %d threads.\n", nthreads);

    const int TIME_DIVISION = 100;
    const int division = (int)(TIME_DIVISION * (tf - ti)) > TIME_DIVISION ? (int)(TIME_DIVISION * (tf - ti)) : TIME_DIVISION;
    const double h1 = (abs(tf - ti)) / division;

    vector<double> Time(division + 1);

    for (int i = 0; i <= division; i++)
    {
        Time[i] = ti + (i * h1);
    }

    std::cout << "calculations started for****************" << endl;

    std::cout << "hbar = " << hbar << endl;
    std::cout << "alpha = " << alpha << endl;
    std::cout << "gama = " << gama << endl;
    std::cout << "w = " << w << endl;
    std::cout << "a = " << a << endl;
    std::cout << "q = " << q << endl;
    std::cout << "ef = " << ef << endl;
    std::cout << "dephase = " << dephase << endl;
    std::cout << "NX = " << NX << endl;
    std::cout << "NY = " << NY << endl;
    std::cout << "Nx = " << Nx << endl;
    std::cout << "Ny = " << Ny << endl;
    std::cout << "lower_limitX = " << lower_limitX << endl;
    std::cout << "upper_limitX = " << upper_limitX << endl;
    std::cout << "lower_limitY = " << lower_limitY << endl;
    std::cout << "upper_limitY = " << upper_limitY << endl;

    std::cout << "A0X = " << A0X << endl;
    if (A0X != 0.0)
    {
        std::cout << "ETPX = " << ETPX << endl;
        std::cout << "delayX = " << delayX << endl;
        if (pulse_wrapperX == "envelope")
        {
            std::cout << "pulse_typeX = " << pulse_typeX << " with envelope" << endl;
            if (pulse_typeX == "gaussian")
            {
                std::cout << "nuX = " << nuX << endl;
                std::cout << "sigmaX = " << sigmaX << endl;
                std::cout << "ntt_sigmaX = " << ntt_sigmaX << endl;
            }
            else if (pulse_typeX == "sin^2")
            {
                std::cout << "nuX = " << nuX << endl;
                std::cout << "number_of_cycleX = " << number_of_cycleX << endl;
            }
        }
        else if (pulse_wrapperX == "NA")
        {
            std::cout << " pulse_typeX = " << pulse_typeX << "without envelope" << endl;
            if (pulse_typeX == "gaussian")
            {
                std::cout << "sigmaX = " << sigmaX << endl;
                std::cout << "ntt_sigmaX = " << ntt_sigmaX << endl;
            }
            else if (pulse_typeX == "sin^2")
            {
                std::cout << "number_of_cycleX = " << number_of_cycleX << endl;
                std::cout << "nuX = " << nuX << endl;
            }
            else if (pulse_typeX == "sin" || pulse_typeX == "cos")
            {
                std::cout << "nuX = " << nuX << endl;
            }
            else if (pulse_typeX == "linear")
            {
                std::cout << "durationX = " << durationX << endl;
            }
        }
    }

    std::cout << "A0Y = " << A0Y << endl;
    if (A0Y != 0.0)
    {
        std::cout << "ETPY = " << ETPY << endl;
        std::cout << "delayY = " << delayY << endl;
        if (pulse_wrapperY == "envelope")
        {
            std::cout << "pulse_typeY = " << pulse_typeY << " with envelope" << endl;
            if (pulse_typeY == "gaussian")
            {
                std::cout << "nuY = " << nuY << endl;
                std::cout << "sigmaY = " << sigmaY << endl;
                std::cout << "ntt_sigmaY = " << ntt_sigmaY << endl;
            }
            else if (pulse_typeY == "sin^2")
            {
                std::cout << "nuY = " << nuY << endl;
                std::cout << "number_of_cycleY = " << number_of_cycleY << endl;
            }
        }
        else if (pulse_wrapperY == "NA")
        {
            std::cout << " pulse_typeY = " << pulse_typeY << "without envelope" << endl;
            if (pulse_typeY == "gaussian")
            {
                std::cout << "sigmaY = " << sigmaY << endl;
                std::cout << "ntt_sigmaY = " << ntt_sigmaY << endl;
            }
            else if (pulse_typeY == "sin^2")
            {
                std::cout << "number_of_cycleY = " << number_of_cycleY << endl;
                std::cout << "nuY = " << nuY << endl;
            }
            else if (pulse_typeY == "sin" || pulse_typeY == "cos")
            {
                std::cout << "nuY = " << nuY << endl;
            }
            else if (pulse_typeY == "linear")
            {
                std::cout << "durationY = " << durationY << endl;
            }
        }
    }
    std::cout << "initial time = " << ti << endl;
    std::cout << "final time = " << tf << endl;

    std::cout << "************************************************************************************************" << endl;

    printCurrentDateTime();
    const auto start = chrono::high_resolution_clock::now();

    try
    {
        std::cout << "started coordinates" << endl;
        coordinates(ti, lower_limitX, upper_limitX, lower_limitY, upper_limitY);
        auto end = chrono::high_resolution_clock::now();
        double elapsed_time = chrono::duration<double>(end - start).count();
        std::cout << "Elapsed time in coordinates: " << elapsed_time << " seconds." << endl;
    }
    catch (const exception &e)
    {
        cerr << setprecision(precision) << "Exception: " + string(e.what()) << endl;
        auto end = chrono::high_resolution_clock::now();
        double elapsed_time = chrono::duration<double>(end - start).count();
        std::cout << "Elapsed time: " << elapsed_time << " seconds." << endl;
        return 1;
    }
    catch (...)
    {
        cerr << "unknown error occured" << endl;
        auto end = chrono::high_resolution_clock::now();
        double elapsed_time = chrono::duration<double>(end - start).count();
        std::cout << "Elapsed time: " << elapsed_time << " seconds." << endl;
        return 1;
    }

    std::cout << "************************************************************************************************" << endl;
    const int integral_divisionX = NX / Nx;
    const int integral_divisionY = NY / Ny;

    std::cout << "integral_divisionX = " << integral_divisionX << endl;
    std::cout << "integral_divisionY = " << integral_divisionY << endl;

    const double DX = abs(upper_limitX - lower_limitX) / integral_divisionX;
    const double DY = abs(upper_limitY - lower_limitY) / integral_divisionY;

    vector<double> JX(Time.size(), 0.0);
    vector<double> JY(Time.size(), 0.0);

    std::cout << "Total number of iterations = " << static_cast<int>(integral_divisionX * integral_divisionY) << endl;
    int iteration = 0;
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
            try
            {
                iteration++;
                std::cout << "*************started J calculation iteration = " << iteration << "***************" << endl;
                integration(llX, ulX, llY, ulY, Time, jx, jy, JX_temp, JY_temp);
                auto end = chrono::high_resolution_clock::now();
                double elapsed_time = chrono::duration<double>(end - start).count();
                std::cout << setprecision(precision) << "Elapsed time of J calculation : " << elapsed_time << " seconds" << endl;
            }
            catch (const exception &e)
            {
                cerr << setprecision(precision) << "Exception : " + string(e.what()) << endl;
                auto end = chrono::high_resolution_clock::now();
                double elapsed_time = chrono::duration<double>(end - start).count();
                std::cout << setprecision(precision) << "Elapsed time : " << elapsed_time << " seconds." << endl;
                return 1;
            }
            transform(JX_temp.begin(), JX_temp.end(), JX.begin(), JX.begin(), plus<double>());
            transform(JY_temp.begin(), JY_temp.end(), JY.begin(), JY.begin(), plus<double>());
        }
    }

    ofstream file;

    file.open("./data.csv");

    if (file.is_open())
    {
        file << setprecision(precision);
        const size_t n_pos = P_x.size();
        const size_t n_time = Time.size();
        const size_t n_rows = std::max(n_pos, n_time);

        file << "px,py,Time,Jx,Jy,Ax,Ay\n";

        for (size_t i = 0; i < n_rows; ++i)
        {
            if (i < n_pos)
                file << P_x[i];
            file << ',';

            if (i < n_pos)
                file << P_y[i];
            file << ',';

            if (i < n_time)
                file << Time[i];
            file << ',';

            if (i < n_time)
                file << (JX[i] / (NX * NY));
            file << ',';

            if (i < n_time)
                file << (JY[i] / (NX * NY));
            file << ',';

            if (i < n_time)
                file << Ax(Time[i]);
            file << ',';

            if (i < n_time)
                file << Ay(Time[i]);

            file << '\n';
        }
    }
    else
    {
        cerr << "Error: could not open file for coordinates" << endl;
        return 1;
    }

    file.close();

    file.open("./parameters.txt");

    if (file.is_open())
    {
        file << "hbar = " << hbar << endl;
        file << "alpha = " << alpha << endl;
        file << "gama = " << gama << endl;
        file << "w = " << w << endl;
        file << "a = " << a << endl;
        file << "q = " << q << endl;
        file << "ef = " << ef << endl;
        file << "dephase = " << dephase << endl;
        file << "NX = " << NX << endl;
        file << "NY = " << NY << endl;
        file << "Nx = " << Nx << endl;
        file << "Ny = " << Ny << endl;
        file << "lower_limitX = " << lower_limitX << endl;
        file << "upper_limitX = " << upper_limitX << endl;
        file << "lower_limitY = " << lower_limitY << endl;
        file << "upper_limitY = " << upper_limitY << endl;

        file << "A0X = " << A0X << endl;
        if (A0X != 0.0)
        {
            file << "ETPX = " << ETPX << endl;
            file << "delayX = " << delayX << endl;
            if (pulse_wrapperX == "envelope")
            {
                file << "pulse_typeX = " << pulse_typeX << " with envelope" << endl;
                if (pulse_typeX == "gaussian")
                {
                    file << "nuX = " << nuX << endl;
                    file << "sigmaX = " << sigmaX << endl;
                    file << "ntt_sigmaX = " << ntt_sigmaX << endl;
                }
                else if (pulse_typeX == "sin^2")
                {
                    file << "nuX = " << nuX << endl;
                    file << "number_of_cycleX = " << number_of_cycleX << endl;
                }
            }
            else if (pulse_wrapperX == "NA")
            {
                file << " pulse_typeX = " << pulse_typeX << "without envelope" << endl;
                if (pulse_typeX == "gaussian")
                {
                    file << "sigmaX = " << sigmaX << endl;
                    file << "ntt_sigmaX = " << ntt_sigmaX << endl;
                }
                else if (pulse_typeX == "sin^2")
                {
                    file << "number_of_cycleX = " << number_of_cycleX << endl;
                    file << "nuX = " << nuX << endl;
                }
                else if (pulse_typeX == "sin" || pulse_typeX == "cos")
                {
                    file << "nuX = " << nuX << endl;
                }
                else if (pulse_typeX == "linear")
                {
                    file << "durationX = " << durationX << endl;
                }
            }
        }

        file << "A0Y = " << A0Y << endl;
        if (A0Y != 0.0)
        {
            file << "ETPY = " << ETPY << endl;
            file << "delayY = " << delayY << endl;
            if (pulse_wrapperY == "envelope")
            {
                file << "pulse_typeY = " << pulse_typeY << " with envelope" << endl;
                if (pulse_typeY == "gaussian")
                {
                    file << "nuY = " << nuY << endl;
                    file << "sigmaY = " << sigmaY << endl;
                    file << "ntt_sigmaY = " << ntt_sigmaY << endl;
                }
                else if (pulse_typeY == "sin^2")
                {
                    file << "nuY = " << nuY << endl;
                    file << "number_of_cycleY = " << number_of_cycleY << endl;
                }
            }
            else if (pulse_wrapperY == "NA")
            {
                file << " pulse_typeY = " << pulse_typeY << "without envelope" << endl;
                if (pulse_typeY == "gaussian")
                {
                    file << "sigmaY = " << sigmaY << endl;
                    file << "ntt_sigmaY = " << ntt_sigmaY << endl;
                }
                else if (pulse_typeY == "sin^2")
                {
                    file << "number_of_cycleY = " << number_of_cycleY << endl;
                    file << "nuY = " << nuY << endl;
                }
                else if (pulse_typeY == "sin" || pulse_typeY == "cos")
                {
                    file << "nuY = " << nuY << endl;
                }
                else if (pulse_typeY == "linear")
                {
                    file << "durationY = " << durationY << endl;
                }
            }
        }

        file << "initial time = " << ti << endl;
        file << "final time = " << tf << endl;
    }
    else
    {
        throw runtime_error("Error: could not open file for parameters");
    }
    file.close();

    std::cout << "calculations ended ****************" << endl;

    printCurrentDateTime();
    auto end = chrono::high_resolution_clock::now();

    double elapsed_time = chrono::duration<double>(end - start).count();

    std::cout << "Elapsed time: " << elapsed_time << " seconds." << endl;

    return 0;
}