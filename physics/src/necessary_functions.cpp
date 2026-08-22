// necessary_functions.cpp

#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include "necessary_functions.h"

// ═══════════════════════════════════════════════════════ task.csv reading ══
// See the header for why this exists and why every column is required.

PulseKind pulse_kind_from_string(const std::string &text)
{
    // Kept in one-to-one correspondence with operator<<(std::ostream&,
    // PulseKind) by construction: these are its output spellings, and the
    // writer puts exactly those in the cell, so a task.csv value and a
    // parameters.txt line read the same and no translation table is needed.
    if (text == "NA")
        return PulseKind::NA;
    if (text == "Envelope")
        return PulseKind::Envelope;
    if (text == "Gaussian")
        return PulseKind::Gaussian;
    if (text == "sin^2")
        return PulseKind::Sin2;
    if (text == "Linear")
        return PulseKind::Linear;
    if (text == "sin")
        return PulseKind::Sin;
    if (text == "cos")
        return PulseKind::Cos;

    throw std::runtime_error(
        "'" + text + "' is not a pulse kind. Valid spellings are exactly what "
        "operator<< prints: NA, Envelope, Gaussian, sin^2, Linear, sin, cos.");
}

std::string trim_cell(const std::string &text)
{
    const auto is_space = [](unsigned char c)
    { return std::isspace(c) != 0; };

    auto begin = std::find_if_not(text.begin(), text.end(), is_space);
    auto end = std::find_if_not(text.rbegin(), text.rend(), is_space).base();
    return (begin < end) ? std::string(begin, end) : std::string();
}

std::vector<std::string> split_csv(const std::string &line)
{
    std::vector<std::string> cells;
    std::string cell;
    std::istringstream stream(line);
    while (std::getline(stream, cell, ','))
        cells.push_back(trim_cell(cell));
    return cells;
}

static std::string join(const std::vector<std::string> &items)
{
    std::string out;
    for (size_t i = 0; i < items.size(); ++i)
        out += (i ? ", " : "") + items[i];
    return out;
}

void require_known_columns(const std::string &path,
                           const std::vector<std::string> &header,
                           const std::vector<std::string> &known)
{
    std::vector<std::string> unknown;
    for (const auto &column : header)
        if (std::find(known.begin(), known.end(), column) == known.end())
            unknown.push_back(column);

    if (!unknown.empty())
        throw std::runtime_error(
            path + ": column(s) [" + join(unknown) + "] are not read by this "
            "binary. An unread column is silently ignored, so the parameter it "
            "was meant to set would keep an unchosen value and the run would "
            "look fine. Columns this binary reads: [" + join(known) + "].");
}

void require_columns(const std::string &path,
                     const std::vector<std::string> &header,
                     const std::vector<std::string> &required)
{
    std::vector<std::string> missing;
    for (const auto &column : required)
        if (std::find(header.begin(), header.end(), column) == header.end())
            missing.push_back(column);

    if (!missing.empty())
        throw std::runtime_error(
            path + ": missing required column(s) [" + join(missing) + "]. "
            "Every column is required -- this file is always machine-written, "
            "so a missing one is a bug, and there are deliberately no defaults "
            "to fall back on.");
}

TaskRow::TaskRow(std::string path, std::vector<std::string> header,
                 std::vector<std::string> cells)
    : path_(std::move(path)), header_(std::move(header)), cells_(std::move(cells))
{
    if (header_.size() != cells_.size())
        throw std::runtime_error(
            path_ + ": the header has " + std::to_string(header_.size()) +
            " column(s) but the data row has " + std::to_string(cells_.size()) +
            ". A cell most likely contains a comma; this format has no quoting.");
}

const std::string &TaskRow::cell(const std::string &key) const
{
    for (size_t i = 0; i < header_.size(); ++i)
        if (header_[i] == key)
            return cells_[i];
    throw std::runtime_error(path_ + ": no column '" + key + "'");
}

double TaskRow::dbl(const std::string &key) const
{
    const std::string &value = cell(key);
    size_t consumed = 0;
    double out = 0.0;
    try
    {
        out = std::stod(value, &consumed);
    }
    catch (const std::exception &)
    {
        throw std::runtime_error(path_ + ": column '" + key + "' = '" + value +
                                 "' is not a number");
    }
    if (consumed != value.size())
        throw std::runtime_error(
            path_ + ": column '" + key + "' = '" + value + "' has trailing "
            "characters after the number; refusing to use the prefix.");
    return out;
}

int TaskRow::integer(const std::string &key) const
{
    const std::string &value = cell(key);
    size_t consumed = 0;
    int out = 0;
    try
    {
        out = std::stoi(value, &consumed);
    }
    catch (const std::exception &)
    {
        throw std::runtime_error(path_ + ": column '" + key + "' = '" + value +
                                 "' is not an integer");
    }
    if (consumed != value.size())
        // std::stoi("10.5") returns 10 and stops at the '.', silently. That
        // is a different parameter point from the one that was requested.
        throw std::runtime_error(
            path_ + ": column '" + key + "' = '" + value + "' is not a whole "
            "number; refusing to truncate it.");
    return out;
}

std::string TaskRow::text(const std::string &key) const { return cell(key); }

PulseKind TaskRow::kind(const std::string &key) const
{
    try
    {
        return pulse_kind_from_string(cell(key));
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(path_ + ": column '" + key + "': " + e.what());
    }
}

TaskRow read_single_task_row(const std::string &path,
                             const std::vector<std::string> &known)
{
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("cannot open task file '" + path + "'");

    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(file, line))
    {
        const std::string trimmed = trim_cell(line);
        if (trimmed.empty() || trimmed[0] == '#')
            continue;
        if (header.empty())
            header = split_csv(trimmed);
        else
            rows.push_back(split_csv(trimmed));
    }

    if (header.empty())
        throw std::runtime_error(path + ": file is empty (no header row)");

    require_known_columns(path, header, known);
    require_columns(path, header, known);

    if (rows.size() != 1)
        throw std::runtime_error(
            path + ": found " + std::to_string(rows.size()) + " data row(s), "
            "expected exactly 1. In this project ONE parameter point expands "
            "into an N1*N2 k-grid that ALL MPI ranks and ALL OpenMP threads "
            "cooperate on -- the k-list is block-decomposed across ranks and "
            "reduced once at the end (see ChargeCurrent). A row is therefore a "
            "whole job, not a unit of work to pack alongside others. For N "
            "points, write N run directories.");

    return TaskRow(path, header, rows.front());
}

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
