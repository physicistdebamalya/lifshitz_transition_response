// pulse_type.cpp

#include <iomanip>
#include "./pulse_type.h"

PulseType::PulseType(const std::string pulse_type, const std::string pulse_wrapper, const double A0, const double sigma, const double ntt_sigma, const double nu, const double ETP, const double ti, const int number_of_cycle_of_pulse, const double duration)
{
    this->pulse_type = pulse_type;
    this->pulse_wrapper = pulse_wrapper;
    this->A0 = A0;
    this->sigma = sigma;
    this->omega = 2 * PI * nu;
    this->ti = ti;
    this->T_pulse = 1.0 / nu;
    this->number_of_cycle_of_pulse = number_of_cycle_of_pulse;

    if (pulse_type == "gaussian")
    {
        this->ttotal = ntt_sigma * sigma;
    }
    else if (pulse_type == "sin^2")
    {
        this->ttotal = number_of_cycle_of_pulse * T_pulse;
    }
    else if (pulse_type == "sin" || pulse_type == "cos")
    {
        this->ttotal = T_pulse;
    }
    else if (pulse_type == "linear")
    {
        this->ttotal = duration;
    }
    else
    {
        throw std::invalid_argument("The pulse type is not supported.");
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
        A = A0 * exp(-(1.0 / 2.0) * pow((t2 - mu) / sigma, 2));
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
        A = A0 * pow(sin((omega / (2 * number_of_cycle_of_pulse)) * t2), 2);
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
    A = gaussian(t) * sin(omega * t2);
    return A;
}
double PulseType::sin_envelope(const double t) const
{
    double A;
    double t2 = t - ti;
    A = sin2(t) * sin(omega * t2);
    return A;
}
double PulseType::Sin(const double t) const
{
    double A;
    double t2 = t - ti;
    A = sin(omega * t2);
    return A;
}
double PulseType::Cos(const double t) const
{
    double A;
    double t2 = t - ti;
    A = cos(omega * t2);
    return A;
}

double PulseType::PULSE(const double t) const
{
    if (pulse_wrapper == "envelope")
    {
        if (pulse_type == "gaussian")
        {
            return gaussian_envelope(t);
        }
        else if (pulse_type == "sin^2")
        {
            return sin_envelope(t);
        }
        else
        {
            throw std::invalid_argument("The pulse type is not supported.");
        }
    }
    else if (pulse_wrapper == "NA")
    {
        if (pulse_type == "gaussian")
        {
            return gaussian(t);
        }
        else if (pulse_type == "sin^2")
        {
            return sin2(t);
        }
        else if (pulse_type == "sin")
        {
            return Sin(t);
        }
        else if (pulse_type == "cos")
        {
            return Cos(t);
        }
        else if (pulse_type == "linear")
        {
            return (A0 * t);
        }
        else
        {
            throw std::invalid_argument("The pulse type is not supported.");
        }
    }
    else
    {
        throw std::invalid_argument("The pulse wrapper is not supported.");
    }
}
