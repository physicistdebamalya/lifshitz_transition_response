// necessary_functions.cpp

#include <iomanip>
#include "necessary_functions.h"

namespace Pauli
{
    const Eigen::Matrix2cd sx = (Eigen::Matrix2cd() << 0, 1.0, 1.0, 0).finished();

    const Eigen::Matrix2cd sy = (Eigen::Matrix2cd() << 0, -constants::z, constants::z, 0).finished();

    const Eigen::Matrix2cd sz = (Eigen::Matrix2cd() << 1.0, 0, 0, -1.0).finished();
}

ProgramTimer::ProgramTimer()
{
    start_time = std::chrono::steady_clock::now();
    elapsed_time = 0.0;
}

void ProgramTimer::stop()
{
    end_time = std::chrono::steady_clock::now();
    elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
}

std::string ProgramTimer::formatted() const
{
    auto dur = std::chrono::duration<double>(elapsed_time);

    auto h = std::chrono::duration_cast<std::chrono::hours>(dur);
    dur -= h;

    auto m = std::chrono::duration_cast<std::chrono::minutes>(dur);
    dur -= m;

    std::stringstream ss;

    const auto hours_count = h.count();

    if (hours_count >= 24)
    {
        long days = static_cast<long>(hours_count / 24);
        long rem_hours = static_cast<long>(hours_count % 24);

        ss << days << " d ";
        if (rem_hours > 0)
            ss << rem_hours << " hr ";
    }
    else if (hours_count > 0)
    {
        ss << hours_count << " hr ";
    }

    if (m.count() > 0)
    {
        ss << m.count() << " min ";
    }

    ss << std::fixed << std::setprecision(3) << dur.count() << " s";

    return ss.str();
}

void printCurrentDateTime()
{
    // Get current time as a time_point object
    auto now = std::chrono::system_clock::now();

    // Convert time_point object to time_t
    time_t currentTime = std::chrono::system_clock::to_time_t(now);

    // Convert time_t to tm struct
    tm *timeStruct = localtime(&currentTime);

    // Print date and time
    std::cout << "date: " << std::put_time(timeStruct, "%d-%m-%Y") << ", time: " << std::put_time(timeStruct, "%H-%M-%S") << std::endl;
}

// full constructor
PulseType::PulseType(const PulseKind pulse_type_, const PulseKind pulse_wrapper_, const double A0, const double sigma,
                     const double ntt_sigma, const double nu, const double ETP, const double ti, const int number_of_cycle_of_pulse, const double duration)
{
    this->pulse_type = pulse_type_;
    this->pulse_wrapper = pulse_wrapper_;
    this->A0 = A0;
    this->sigma = sigma;
    this->omega = 2 * constants::PI * nu;
    this->ti = ti;
    this->T_pulse = 1.0 / nu;
    this->number_of_cycle_of_pulse = number_of_cycle_of_pulse;
    this->nu = nu;                 // retained for printPulseParams(); see the
    this->ntt_sigma = ntt_sigma;   // header comment on these three members
    this->duration = duration;

    switch (pulse_type)
    {
    case PulseKind::Gaussian:
        this->ttotal = ntt_sigma * sigma;
        break;
    case PulseKind::Sin2:
        this->ttotal = number_of_cycle_of_pulse * T_pulse;
        break;
    case PulseKind::Sin:
    case PulseKind::Cos:
        this->ttotal = T_pulse;
        break;
    case PulseKind::Linear:
        this->ttotal = duration;
        break;
    default:
        throw std::invalid_argument("The pulse_type is not supported.");
    }

    this->tf = ti + (ttotal * (1.0 + ETP / 100.0));
}

double PulseType::gaussian(const double t) const
{
    double A;
    const double mu = ttotal / 2.0;
    double t2 = t - ti;
    if ((t2 >= 0) && (t2 <= ttotal))
    {
        const double u = (t2 - mu) / sigma;
        A = A0 * std::exp(-(1.0 / 2.0) * (u * u));
    }
    else
    {
        A = 0.0;
    }
    return A;
}

double PulseType::sin2(const double t) const
{
    double A;
    double t2 = t - ti;
    if ((t2 >= 0) && (t2 <= ttotal))
    {
        const double s = std::sin((omega / (2 * number_of_cycle_of_pulse)) * t2);
        A = A0 * (s * s);
    }
    else
    {
        A = 0.0;
    }
    return A;
}

double PulseType::gaussian_envelope(const double t) const
{
    double A;
    double t2 = t - ti;
    A = gaussian(t) * std::sin(omega * t2);
    return A;
}
double PulseType::sin_envelope(const double t) const
{
    double A;
    double t2 = t - ti;
    A = sin2(t) * std::sin(omega * t2);
    return A;
}
double PulseType::Sin(const double t) const
{
    double A;
    double t2 = t - ti;
    A = std::sin(omega * t2);
    return A;
}
double PulseType::Cos(const double t) const
{
    double A;
    double t2 = t - ti;
    A = std::cos(omega * t2);
    return A;
}

double PulseType::PULSE(const double t) const
{
    switch (pulse_wrapper)
    {
    case PulseKind::Envelope:
        switch (pulse_type)
        {
        case PulseKind::Gaussian:
            return gaussian_envelope(t);
        case PulseKind::Sin2:
            return sin_envelope(t);
        default:
            throw std::invalid_argument("The pulse type is not supported.");
        }
        break;
    case PulseKind::NA:
        switch (pulse_type)
        {
        case PulseKind::Gaussian:
            return gaussian(t);
        case PulseKind::Sin2:
            return sin2(t);
        case PulseKind::Sin:
            return Sin(t);
        case PulseKind::Cos:
            return Cos(t);
        case PulseKind::Linear:
            return (A0 * t);
        default:
            throw std::invalid_argument("The pulse type is not supported.");
        }
    default:
        throw std::invalid_argument("The pulse wrapper is not supported.");
    }
}

// Ported from Superconductivity's PulseType::printPulseParams
// (necessary_functions.cpp there), with the phi/carrier-phase branch
// removed: this project's pulse model has no phase parameter to report.
//
// Category-C emission, mirroring analysis/snippets.py's emit_pulse(): A0 is
// always written -- it is the channel on/off switch -- and if it is zero,
// nothing else about that channel means anything, so nothing else is
// written. Two models call this identically (see two_band.cpp/four_band.cpp
// main()), which is the point: this conditional structure previously lived
// twice, hand-duplicated, inside each model's print_core_parameters lambda.
void PulseType::printPulseParams(std::ostream &os, const std::string &axis) const
{
    os << "A0" << axis << " = " << A0 << "\n";
    if (A0 == 0.0)
        return;

    os << "delay" << axis << " = " << ti << "\n";

    if (pulse_wrapper == PulseKind::Envelope)
    {
        os << "pulse_type" << axis << " = " << pulse_type << " with Envelope" << "\n";
        if (pulse_type == PulseKind::Gaussian)
        {
            os << "nu" << axis << " = " << nu << "\n";
            os << "sigma" << axis << " = " << sigma << "\n";
            os << "ntt_sigma" << axis << " = " << ntt_sigma << "\n";
        }
        else if (pulse_type == PulseKind::Sin2)
        {
            os << "nu" << axis << " = " << nu << "\n";
            os << "number_of_cycle" << axis << " = " << number_of_cycle_of_pulse << "\n";
        }
    }
    else if (pulse_wrapper == PulseKind::NA)
    {
        os << "pulse_type" << axis << " = " << pulse_type << " without Envelope" << "\n";
        if (pulse_type == PulseKind::Gaussian)
        {
            os << "sigma" << axis << " = " << sigma << "\n";
            os << "ntt_sigma" << axis << " = " << ntt_sigma << "\n";
        }
        else if (pulse_type == PulseKind::Sin2)
        {
            os << "number_of_cycle" << axis << " = " << number_of_cycle_of_pulse << "\n";
            os << "nu" << axis << " = " << nu << "\n";
        }
        else if (pulse_type == PulseKind::Sin || pulse_type == PulseKind::Cos)
        {
            os << "nu" << axis << " = " << nu << "\n";
        }
        else if (pulse_type == PulseKind::Linear)
        {
            os << "duration" << axis << " = " << duration << "\n";
        }
    }
}
