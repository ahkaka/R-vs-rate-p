#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <random>
#include <numeric>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>

// ========= 类型别名和常量 =========
using cplx = std::complex<double>;
const double PI = M_PI;
const cplx I(0.0, 1.0);
thread_local std::mt19937 gen;

// ========= 辅助函数 (未改变) =========
/**
 * @brief 生成一个 [start, end] 范围内的对数间隔向量
 */
std::vector<double> logspace(double start, double end, int steps) {
    if (start <= 0.0 || end <= 0.0) {
        std::cerr << "错误: logspace 的起始值和结束值必须大于零!" << std::endl;
        return std::vector<double>(steps, 0.0);
    }
    std::vector<double> vec(steps);
    // 使用自然对数进行间隔计算
    double log_start = std::log(start);
    double log_end = std::log(end);
    double log_step_size = (log_end - log_start) / (steps - 1);

    for (int i = 0; i < steps; ++i) {
        double current_log = log_start + i * log_step_size;
        vec[i] = std::exp(current_log); // 转换回线性尺度
    }
    return vec;
}
std::vector<double> sample_lorentzian(double gamma, size_t size) {
    std::cauchy_distribution<double> dist(0.0, gamma);
    std::vector<double> samples(size);
    for (size_t i = 0; i < size; ++i) {
        samples[i] = dist(gen);
    }
    return samples;
}

cplx calculate_Z(const std::vector<double>& thetas) {
    if (thetas.empty()) return 0.0;
    cplx z_sum = 0.0;
    for (double theta : thetas) {
        z_sum += std::exp(I * theta);
    }
    return z_sum / static_cast<double>(thetas.size());
}

std::vector<double> linspace(double min, double max, int steps) {
    std::vector<double> vec(steps);
    double step_size = (max - min) / (steps - 1);
    for (int i = 0; i < steps; ++i) {
        vec[i] = min + i * step_size;
    }
    return vec;
}

// ========= 核心物理逻辑 (未改变) =========

void heun_step_thetas(std::vector<double>& thetas,
                      const std::vector<double>& omegas,
                      const std::vector<double>& Ks,
                      double dt) {
    size_t N = thetas.size();
    std::vector<double> theta_dot(N);
    std::vector<double> thetas_pred(N);
    std::vector<double> theta_dot_pred(N);
    cplx Z = calculate_Z(thetas);
    double R = std::abs(Z);
    double phi = std::arg(Z);
    for (size_t i = 0; i < N; ++i) {
        theta_dot[i] = omegas[i] + Ks[i] * R * std::sin(phi - thetas[i]);
    }
    for (size_t i = 0; i < N; ++i) {
        thetas_pred[i] = thetas[i] + dt * theta_dot[i];
    }
    cplx Z_pred = calculate_Z(thetas_pred);
    double R_pred = std::abs(Z_pred);
    double phi_pred = std::arg(Z_pred);
    for (size_t i = 0; i < N; ++i) {
        theta_dot_pred[i] = omegas[i] + Ks[i] * R_pred * std::sin(phi_pred - thetas_pred[i]);
    }
    for (size_t i = 0; i < N; ++i) {
        thetas[i] = thetas[i] + dt * 0.5 * (theta_dot[i] + theta_dot_pred[i]);
    }
}

void chemical_reaction_tau_leaping(std::vector<bool>& types,
                                   std::vector<double>& Ks,
                                   double K1, double K2,
                                   double r1, double r2, double dt) {
    size_t N = types.size();
    size_t n_positive = std::count_if(types.begin(), types.end(), [](bool b){ return b; });
    size_t n_negative = N - n_positive;
    std::poisson_distribution<int> poiss_pos_to_neg(r1 * n_positive * dt);
    std::poisson_distribution<int> poiss_neg_to_pos(r2 * n_negative * dt);
    int n_pos_to_neg = poiss_pos_to_neg(gen);
    int n_neg_to_pos = poiss_neg_to_pos(gen);
    n_pos_to_neg = std::min(n_pos_to_neg, static_cast<int>(n_positive));
    n_neg_to_pos = std::min(n_neg_to_pos, static_cast<int>(n_negative));
    if (n_pos_to_neg > 0) {
        std::vector<size_t> pos_indices;
        for (size_t i = 0; i < N; ++i) { if (types[i]) pos_indices.push_back(i); }
        std::shuffle(pos_indices.begin(), pos_indices.end(), gen);
        for (int i = 0; i < n_pos_to_neg; ++i) {
            size_t idx = pos_indices[i];
            types[idx] = false;
            Ks[idx] = K1;
        }
    }
    if (n_neg_to_pos > 0) {
        std::vector<size_t> neg_indices;
        for (size_t i = 0; i < N; ++i) { if (!types[i]) neg_indices.push_back(i); }
        std::shuffle(neg_indices.begin(), neg_indices.end(), gen);
        for (int i = 0; i < n_neg_to_pos; ++i) {
            size_t idx = neg_indices[i];
            types[idx] = true;
            Ks[idx] = K2;
        }
    }
}

// ========= !!! 修改后的模拟函数 !!! =========

/**
 * @brief 运行单次微观模拟，带自适应收敛检查
 * @param sim_result (out) 用于返回最终的 R 值
 * @return (bool) 如果模拟是提前收敛的，则返回 true
 */
bool run_single_simulation(
    // 基本参数
    int N, double K1, double K2, double p0, double gamma,
    double r1, double r2, double dt, double t_max, unsigned int seed,
    // 收敛参数
    double min_transient_time, double check_block_duration, double convergence_tol,
    // 输出参数
    double& sim_result)
{
    gen.seed(seed);
    
    // --- 1. 初始化 ---
    std::vector<bool> types(N);
    std::vector<double> Ks(N);
    std::vector<double> thetas(N);
    std::uniform_real_distribution<double> unif_prob(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        types[i] = unif_prob(gen) < p0;
        Ks[i] = types[i] ? K2 : K1;
        thetas[i] = unif_prob(gen) * 2.0 * PI;
    }
    std::vector<double> omegas = sample_lorentzian(gamma, N);

    // --- 2. 转换时间为步数 ---
    int num_steps_total = static_cast<int>(t_max / dt);
    int transient_steps = static_cast<int>(min_transient_time / dt);
    int block_steps = static_cast<int>(check_block_duration / dt);

    if (block_steps == 0) {
         // 错误：块时长太短
         sim_result = -1.0; // 返回一个无效值
         return false;
    }

    // --- 3. 模拟循环与收敛检查 ---
    double mean_prev = 0.0, mean_curr = 0.0;
    double R_block_sum = 0.0;
    int R_block_count = 0;
    bool converged = false;
    bool first_block_calculated = false;

    for (int i = 0; i < num_steps_total; ++i) {
        if (i > 0) {
            chemical_reaction_tau_leaping(types, Ks, K1, K2, r1, r2, dt);
        }

        cplx Z = calculate_Z(thetas);
        heun_step_thetas(thetas, omegas, Ks, dt);

        // --- 收敛检查逻辑 ---
        if (i < transient_steps) {
            // 阶段 1: 暂态，不做任何事
            continue;
        }
        
        // 阶段 2: 累积块数据
        R_block_sum += std::abs(Z);
        R_block_count++;

        // 检查是否完成一个块
        if (R_block_count == block_steps) {
            mean_curr = R_block_sum / R_block_count;

            if (first_block_calculated) {
                // 这不是第一个块，可以进行比较
                if (std::abs(mean_curr - mean_prev) < convergence_tol) {
                    converged = true;
                    break; // 提前退出循环
                }
            }
            
            // 准备下一个块
            mean_prev = mean_curr;
            first_block_calculated = true;
            R_block_sum = 0.0;
            R_block_count = 0;
        }
    } // 结束主模拟循环

    // --- 4. 返回结果 ---
    if (converged) {
        sim_result = mean_curr; // 成功收敛，返回最后稳定的平均值
        return true;
    } else {
        // 未收敛，运行到了 t_max
        if (R_block_count > 0) {
            // 最后一个块未满，计算这个不完整块的平均值
            sim_result = R_block_sum / R_block_count;
        } else {
            // 恰好在块边界上结束
            sim_result = mean_prev; // 返回最后一个完整块的平均值
        }
        return false;
    }
}

// ========= 辅助函数 (未改变) =========
void save_results_to_csv(
    const std::vector<std::vector<double>>& grid,
    const std::vector<double>& x_axis, // p_s
    const std::vector<double>& y_axis  // rate_scale
) {
    std::ofstream outfile("scan_results.csv");
    if (!outfile.is_open()) {
        std::cerr << "错误: 无法打开文件 scan_results.csv 进行写入！" << std::endl;
        return;
    }
    outfile << "rate_scale(Y)";
    for (double p_s : x_axis) { outfile << "," << p_s; }
    outfile << "\n";
    for (size_t i = 0; i < y_axis.size(); ++i) {
        outfile << y_axis[i]; 
        for (size_t j = 0; j < x_axis.size(); ++j) {
            outfile << "," << grid[i][j];
        }
        outfile << "\n";
    }
    outfile.close();
    std::cout << "数据已成功保存到 scan_results.csv" << std::endl;
}

// ========= !!! 修改后的 Main 函数 !!! =========

int main() {
    // --- 1. 定义固定的全局参数 ---
    const int N = 30000;
    const double K1 = -0.5;
    const double K2 = 1.0;
    const double gamma = 0.05;
    const double dt = 0.02;
    const double t_max = 300.0; // 仍然是最大安全时间
    const unsigned int seed_base = 42;

    // --- 2. 定义扫描参数范围 ---
    const int P_STEPS = 30; // X轴 (p_s)
    const int R_STEPS = 30; // Y轴 (rate_scale)

    std::vector<double> p_s_values = linspace(0.3, 0.95, P_STEPS);
    std::vector<double> rate_scale_values = logspace(0.001, 1.0, R_STEPS);

    // --- 3. (新) 定义收敛参数 ---
    const double MIN_TRANSIENT_TIME = 40.0;   // 至少运行这么久才开始检查
    const double CHECK_BLOCK_DURATION = 20.0; // 检查窗口的时长
    const double CONVERGENCE_TOL = 1e-3;      // 收敛容忍度
    
    // 确保 t_max 至少大于暂态时间+一个块
    if (t_max <= MIN_TRANSIENT_TIME + CHECK_BLOCK_DURATION) {
        std::cerr << "警告: t_max 太短，收敛检查可能无法正常工作。" << std::endl;
    }

    // --- 4. 初始化结果存储网格 ---
    std::vector<std::vector<double>> R_steady_state(R_STEPS, std::vector<double>(P_STEPS));

    std::cout << "--- 开始参数扫描 ---" << std::endl;
    std::cout << "参数: N=" << N << ", K1=" << K1 << ", K2=" << K2 << ", t_max=" << t_max << std::endl;
    std::cout << "收敛: t_transient=" << MIN_TRANSIENT_TIME 
              << ", t_block=" << CHECK_BLOCK_DURATION 
              << ", tol=" << CONVERGENCE_TOL << std::endl;
    
    // --- 5. 执行扫描循环 ---
    for (int i = 0; i < R_STEPS; ++i) {
        double rate_scale = rate_scale_values[i];
        
        std::cout << "\n处理 Y 步 " << (i + 1) << "/" << R_STEPS 
                  << " (rate_scale = " << rate_scale << ")" << std::endl;
        std::cout << "[" << std::flush;
        
        for (int j = 0; j < P_STEPS; ++j) {
            double p_s_val = p_s_values[j];
            double r1 = rate_scale;
            double p0 = p_s_val; 
            double r2;
            
            if (p_s_val < (1.0 - 1e-9) && p_s_val > 1e-9) {
                r2 = r1 * p_s_val / (1.0 - p_s_val);
            } else if (p_s_val >= (1.0 - 1e-9)) {
                 r2 = 1e9; // 极大值
            } else {
                 r2 = 0.0; // p_s 几乎为 0
            }

            unsigned int sim_seed = seed_base + i * P_STEPS + j;

            // --- 运行单次模拟 ---
            double R_long_term;
            bool did_converge = run_single_simulation(
                N, K1, K2, p0, gamma, r1, r2, dt, t_max, sim_seed,
                MIN_TRANSIENT_TIME, CHECK_BLOCK_DURATION, CONVERGENCE_TOL,
                R_long_term // 结果通过引用传递回来
            );
            
            R_steady_state[i][j] = R_long_term;

            // 打印进度点 (C=收敛, .=t_max)
            std::cout << (did_converge ? "C" : ".") << std::flush;
        }
        std::cout << "]" << std::endl; // Y 步完成
    }

    std::cout << "\n--- 扫描完成 ---" << std::endl;

    // --- 6. 保存结果 ---
    save_results_to_csv(R_steady_state, p_s_values, rate_scale_values);

    return 0;
}