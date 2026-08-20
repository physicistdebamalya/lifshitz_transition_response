
#pragma once
#include "Eigen/Dense"

constexpr double PI = 3.141592653589793;

class PulseType
{
private:
    std::string pulse_type, pulse_wrapper;
    double A0, sigma, omega, T_pulse, ttotal;
    int number_of_cycle_of_pulse;

    double gaussian(const double) const;
    double gaussian_envelope(const double) const;
    double sin2(const double) const;
    double sin_envelope(const double) const;
    double Sin(const double) const;
    double Cos(const double) const;

public:
    PulseType() {};
    PulseType(const std::string pulse_type, const std::string pulse_wrapper, const double A0, const double sigma, const double ntt_sigma, const double nu, const double ETP, const double ti, const int number_of_cycle_of_pulse, const double duration);
    double PULSE(const double) const;
    double ti, tf;
};

class D2Pulse
{
private:
    const PulseType &pulseX, &pulseY;

public:
    D2Pulse(const PulseType &pulseX, const PulseType &pulseY) : pulseX(pulseX), pulseY(pulseY) {};
    inline Eigen::Vector2d A(double t) const
    {
        Eigen::Vector2d pulse;
        pulse[0] = pulseX.PULSE(t);
        pulse[1] = pulseY.PULSE(t);
        return pulse;
    }
};