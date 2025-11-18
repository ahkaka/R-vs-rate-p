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
#include <omp.h>

// ========= 类型别名和常量 =========
using cplx = std::complex<double>;
const double PI = M_PI;
const cplx I(0.0, 1.0);
thread_local std::mt19937 gen;

// ========= 新增：结果结构体 =========
struct OrderParams {
    double R_total;
    double R_K1;
    double R_K2;
};

// ========= 辅助函数 =========

std::vector<double> logspace(double start, double end, int steps) {
    if (start <= 0.0 || end <= 0.0) return std::vector<double>(steps, 0.0);
    std::vector<double> vec(steps);
    double log_start = std::log(start);
    double log_end = std::log(end);
    double log_step_size = (log_end - log_start) / (steps - 1);
    for (int i = 0; i < steps; ++i) {
        vec[i] = std::exp(log_start + i * log_step_size);
    }
    return vec;
}

std::vector<double> linspace(double min, double max, int steps) {
    std::vector<double> vec(steps);
    double step_size = (max - min) / (steps - 1);
    for (int i = 0; i < steps; ++i) {
        vec[i] = min + i * step_size;
    }
    return vec;
}

std::vector<double> sample_lorentzian(double gamma, size_t size) {
    std::cauchy_distribution<double> dist(0.0, gamma);
    std::vector<double> samples(size);
    for (size_t i = 0; i < size; ++i) samples[i] = dist(gen);
    return samples;
}

// 旧的计算函数（仅用于动力学更新，保持速度）
cplx calculate_Z_fast(const std::vector<double>& thetas) {
    if (thetas.empty()) return 0.0;
    cplx z_sum = 0.0;
    for (double theta : thetas) {
        z_sum += std::exp(I * theta);
    }
    return z_sum / static_cast<double>(thetas.size());
}

// 新的计算函数：同时计算 Total, K1, K2 的序参量（用于数据采集）
OrderParams calculate_all_order_params(const std::vector<double>& thetas, const std::vector<bool>& types) {
    cplx z_sum_total = 0.0;
    cplx z_sum_k1 = 0.0;
    cplx z_sum_k2 = 0.0;
    int count_k1 = 0;
    int count_k2 = 0;
    size_t N = thetas.size();

    for (size_t i = 0; i < N; ++i) {
        cplx z = std::exp(I * thetas[i]);
        z_sum_total += z;
        
        // types[i] == false 对应 K1, true 对应 K2
        if (types[i]) { 
            z_sum_k2 += z;
            count_k2++;
        } else {
            z_sum_k1 += z;
            count_k1++;
        }
    }

    OrderParams res;
    res.R_total = std::abs(z_sum_total) / static_cast<double>(N);
    // 防止除以零（如果某一种群完全消失）
    res.R_K1 = (count_k1 > 0) ? std::abs(z_sum_k1) / count_k1 : 0.0;
    res.R_K2 = (count_k2 > 0) ? std::abs(z_sum_k2) / count_k2 : 0.0;
    
    return res;
}

// ========= 核心物理逻辑 =========

void heun_step_thetas(std::vector<double>& thetas,
                      const std::vector<double>& omegas,
                      const std::vector<double>& Ks,
                      double dt) {
    size_t N = thetas.size();
    std::vector<double> theta_dot(N);
    std::vector<double> thetas_pred(N);
    std::vector<double> theta_dot_pred(N);
    
    // 动力学演化只依赖全局场
    cplx Z = calculate_Z_fast(thetas);
    double R = std::abs(Z);
    double phi = std::arg(Z);
    
    for (size_t i = 0; i < N; ++i) {
        theta_dot[i] = omegas[i] + Ks[i] * R * std::sin(phi - thetas[i]);
    }
    for (size_t i = 0; i < N; ++i) {
        thetas_pred[i] = thetas[i] + dt * theta_dot[i];
    }
    
    cplx Z_pred = calculate_Z_fast(thetas_pred);
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
            types[idx] = false; // 变为 K1
            Ks[idx] = K1;
        }
    }
    if (n_neg_to_pos > 0) {
        std::vector<size_t> neg_indices;
        for (size_t i = 0; i < N; ++i) { if (!types[i]) neg_indices.push_back(i); }
        std::shuffle(neg_indices.begin(), neg_indices.end(), gen);
        for (int i = 0; i < n_neg_to_pos; ++i) {
            size_t idx = neg_indices[i];
            types[idx] = true; // 变为 K2
            Ks[idx] = K2;
        }
    }
}

// ========= 修改后的模拟函数 =========

bool run_single_simulation(
    int N, double K1, double K2, double p0, double gamma,
    double r1, double r2, double dt, double t_max, unsigned int seed,
    double min_transient_time, double check_block_duration, double convergence_tol,
    OrderParams& sim_result // 输出改为结构体引用
) {
    gen.seed(seed);
    
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

    int num_steps_total = static_cast<int>(t_max / dt);
    int transient_steps = static_cast<int>(min_transient_time / dt);
    int block_steps = static_cast<int>(check_block_duration / dt);

    if (block_steps == 0) return false;

    // 收敛检查变量
    double mean_R_total_prev = 0.0;
    double mean_R_total_curr = 0.0;
    
    // 累加变量 (三个参数都需要累加)
    double sum_R_total = 0.0;
    double sum_R_K1 = 0.0;
    double sum_R_K2 = 0.0;
    int block_count = 0;
    
    bool converged = false;
    bool first_block_calculated = false;

    for (int i = 0; i < num_steps_total; ++i) {
        if (i > 0) {
            chemical_reaction_tau_leaping(types, Ks, K1, K2, r1, r2, dt);
        }
        
        heun_step_thetas(thetas, omegas, Ks, dt);

        // --- 收敛检查与数据采集 ---
        if (i < transient_steps) continue;
        
        // 计算当前步的三个序参量
        OrderParams current_params = calculate_all_order_params(thetas, types);
        
        sum_R_total += current_params.R_total;
        sum_R_K1    += current_params.R_K1;
        sum_R_K2    += current_params.R_K2;
        block_count++;

        if (block_count == block_steps) {
            mean_R_total_curr = sum_R_total / block_count;
            
            // 仅使用 Total R 进行收敛判定
            if (first_block_calculated) {
                if (std::abs(mean_R_total_curr - mean_R_total_prev) < convergence_tol) {
                    converged = true;
                    // 计算最终块的平均值作为结果
                    sim_result.R_total = mean_R_total_curr;
                    sim_result.R_K1    = sum_R_K1 / block_count;
                    sim_result.R_K2    = sum_R_K2 / block_count;
                    break; 
                }
            }
            
            mean_R_total_prev = mean_R_total_curr;
            
            // 保存这个块的数据，以防这就是最后一个块但没收敛
            sim_result.R_total = mean_R_total_curr;
            sim_result.R_K1    = sum_R_K1 / block_count;
            sim_result.R_K2    = sum_R_K2 / block_count;

            first_block_calculated = true;
            
            // 重置累加器
            sum_R_total = 0.0;
            sum_R_K1 = 0.0;
            sum_R_K2 = 0.0;
            block_count = 0;
        }
    }

    return converged;
}

// ========= 修改后的保存函数 (支持文件名参数) =========
void save_results_to_csv(
    const std::string& filename,
    const std::vector<std::vector<double>>& grid,
    const std::vector<double>& x_axis, 
    const std::vector<double>& y_axis
) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "错误: 无法打开文件 " << filename << std::endl;
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
    std::cout << "数据已保存到: " << filename << std::endl;
}

// ========= 修改后的 Main 函数 =========

int main() {
    // 1. 定义参数
    const int N = 200;
    const double K1 = -0.5;
    const double K2 = 1.0;
    const double gamma = 0.05;
    const double dt = 0.02;
    const double t_max = 40.0; 
    const unsigned int seed_base = 12345;

    // 增加扫描密度以体现并行优势（可选）
    const int P_STEPS = 25; 
    const int R_STEPS = 25; 

    std::vector<double> p_s_values = linspace(0.3, 0.95, P_STEPS);
    std::vector<double> rate_scale_values = logspace(0.001, 1.0, R_STEPS);

    const double MIN_TRANSIENT_TIME = 40.0;
    const double CHECK_BLOCK_DURATION = 20.0;
    const double CONVERGENCE_TOL = 1e-3;
    
    if (t_max <= MIN_TRANSIENT_TIME + CHECK_BLOCK_DURATION) {
        std::cerr << "警告: t_max 太短，收敛检查可能无法正常工作。" << std::endl;
    }

    // 初始化结果网格
    std::vector<std::vector<double>> grid_total(R_STEPS, std::vector<double>(P_STEPS));
    std::vector<std::vector<double>> grid_k1(R_STEPS, std::vector<double>(P_STEPS));
    std::vector<std::vector<double>> grid_k2(R_STEPS, std::vector<double>(P_STEPS));

    std::cout << "--- 开始参数扫描 (OpenMP 并行版) ---" << std::endl;
    // 获取最大线程数（仅用于显示）
    std::cout << "检测到最大线程数: " << omp_get_max_threads() << std::endl;

    // 外层循环保持串行，以便按行输出进度，看起来更整洁
    for (int i = 0; i < R_STEPS; ++i) {
        double rate_scale = rate_scale_values[i];
        
        std::cout << "\n处理 Y 步 " << (i + 1) << "/" << R_STEPS 
                  << " (rate_scale = " << rate_scale << ")" << std::endl;
        std::cout << "[" << std::flush;
        
        // =========== 并行区域开始 ===========
        // #pragma omp parallel for: 自动将下面的循环分配给多个线程
        // schedule(dynamic): 因为模拟时长不一，动态调度能让快的线程多干活，防止等待
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < P_STEPS; ++j) {
            double p_s_val = p_s_values[j];
            double r1 = rate_scale;
            double p0 = p_s_val; 
            double r2;
            
            if (p_s_val < (1.0 - 1e-9) && p_s_val > 1e-9) {
                r2 = r1 * p_s_val / (1.0 - p_s_val);
            } else if (p_s_val >= (1.0 - 1e-9)) {
                 r2 = 1e9; 
            } else {
                 r2 = 0.0; 
            }

            // 确保种子对每个 (i, j) 都是确定且唯一的
            unsigned int sim_seed = seed_base + i * P_STEPS + j;

            OrderParams result; 
            
            // 每个线程独立运行模拟
            bool did_converge = run_single_simulation(
                N, K1, K2, p0, gamma, r1, r2, dt, t_max, sim_seed,
                MIN_TRANSIENT_TIME, CHECK_BLOCK_DURATION, CONVERGENCE_TOL,
                result 
            );
            
            // 并行写入不同的内存地址，不需要锁
            grid_total[i][j] = result.R_total;
            grid_k1[i][j]    = result.R_K1;
            grid_k2[i][j]    = result.R_K2;

            // 打印进度需要加锁，否则字符会乱成一团
            // critical 块会让线程在这里排队，稍微影响一点点性能，但为了看进度是值得的
            #pragma omp critical 
            {
                std::cout << (did_converge ? "C" : ".") << std::flush;
            }
        }
        // =========== 并行区域结束 ===========

        std::cout << "]" << std::endl; 
    }

    std::cout << "\n--- 扫描完成，正在保存数据 ---" << std::endl;

    save_results_to_csv("scan_R_total.csv", grid_total, p_s_values, rate_scale_values);
    save_results_to_csv("scan_R_K1.csv",    grid_k1,    p_s_values, rate_scale_values);
    save_results_to_csv("scan_R_K2.csv",    grid_k2,    p_s_values, rate_scale_values);

    return 0;
}