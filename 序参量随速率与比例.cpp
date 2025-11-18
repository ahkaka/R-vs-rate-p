#include <iostream>     // 用于控制台输出 (cout, endl)
#include <vector>       // 用于动态数组 (std::vector)
#include <cmath>        // 用于数学函数 (sin, abs, log, M_PI)
#include <complex>      // 用于复数 (std::complex)
#include <random>       // 用于 C++ 随机数生成
#include <numeric>      // 用于 std::accumulate
#include <fstream>      // 用于文件输出 (std::ofstream)
#include <string>       // 用于 std::string
#include <algorithm>    // 用于 std::min, std::shuffle, std::count_if
#include <chrono>       // 用于高精度时钟 (作为随机数种子)

// ========= 类型别名和常量 =========
using cplx = std::complex<double>;
const double PI = M_PI;
// 定义复数单位 i
const cplx I(0.0, 1.0);

// 用于 tau-leaping 中随机选择
// mt19937 是一个高质量的随机数生成器
thread_local std::mt19937 gen;

// ========= 辅助函数 (替换 Numpy) =========

/**
 * @brief 生成洛伦兹(柯西)分布的随机数
 */
std::vector<double> sample_lorentzian(double gamma, size_t size) {
    std::cauchy_distribution<double> dist(0.0, gamma);
    std::vector<double> samples(size);
    for (size_t i = 0; i < size; ++i) {
        samples[i] = dist(gen);
    }
    return samples;
}

/**
 * @brief 计算全局序参量 Z = mean(exp(i * theta))
 */
cplx calculate_Z(const std::vector<double>& thetas) {
    if (thetas.empty()) return 0.0;
    cplx z_sum = 0.0;
    for (double theta : thetas) {
        z_sum += std::exp(I * theta);
    }
    return z_sum / static_cast<double>(thetas.size());
}

/**
 * @brief 生成一个 [min, max] 范围内的线性间隔向量
 */
std::vector<double> linspace(double min, double max, int steps) {
    std::vector<double> vec(steps);
    double step_size = (max - min) / (steps - 1);
    for (int i = 0; i < steps; ++i) {
        vec[i] = min + i * step_size;
    }
    return vec;
}

// ========= 核心物理逻辑 =========

/**
 * @brief 执行一步 Heun 积分
 * @param thetas 相位 (in/out)
 * @param omegas 自然频率 (in)
 * @param Ks 耦合强度 (in)
 * @param dt 时间步长 (in)
 */
void heun_step_thetas(std::vector<double>& thetas,
                      const std::vector<double>& omegas,
                      const std::vector<double>& Ks,
                      double dt) {
    size_t N = thetas.size();
    std::vector<double> theta_dot(N);
    std::vector<double> thetas_pred(N);
    std::vector<double> theta_dot_pred(N);

    // 1. 计算当前导数
    cplx Z = calculate_Z(thetas);
    double R = std::abs(Z);
    double phi = std::arg(Z);
    for (size_t i = 0; i < N; ++i) {
        theta_dot[i] = omegas[i] + Ks[i] * R * std::sin(phi - thetas[i]);
    }

    // 2. 预测步
    for (size_t i = 0; i < N; ++i) {
        thetas_pred[i] = thetas[i] + dt * theta_dot[i];
    }

    // 3. 计算预测点的导数
    cplx Z_pred = calculate_Z(thetas_pred);
    double R_pred = std::abs(Z_pred);
    double phi_pred = std::arg(Z_pred);
    for (size_t i = 0; i < N; ++i) {
        theta_dot_pred[i] = omegas[i] + Ks[i] * R_pred * std::sin(phi_pred - thetas_pred[i]);
    }

    // 4. Heun 修正 (更新相位)
    for (size_t i = 0; i < N; ++i) {
        thetas[i] = thetas[i] + dt * 0.5 * (theta_dot[i] + theta_dot_pred[i]);
    }
}

/**
 * @brief Tau-leaping 算法处理化学反应
 * @param types 个体类型 (in/out) (True=K2, False=K1)
 * @param Ks 耦合强度 (in/out)
 */
void chemical_reaction_tau_leaping(std::vector<bool>& types,
                                   std::vector<double>& Ks,
                                   double K1, double K2,
                                   double r1, double r2, double dt) {
    size_t N = types.size();
    size_t n_positive = std::count_if(types.begin(), types.end(), [](bool b){ return b; });
    size_t n_negative = N - n_positive;

    // 1. 计算泊松事件数 (r1: pos->neg, r2: neg->pos)
    std::poisson_distribution<int> poiss_pos_to_neg(r1 * n_positive * dt);
    std::poisson_distribution<int> poiss_neg_to_pos(r2 * n_negative * dt);

    int n_pos_to_neg = poiss_pos_to_neg(gen);
    int n_neg_to_pos = poiss_neg_to_pos(gen);

    // 2. 限制不超过可用个体
    n_pos_to_neg = std::min(n_pos_to_neg, static_cast<int>(n_positive));
    n_neg_to_pos = std::min(n_neg_to_pos, static_cast<int>(n_negative));

    // 3. 执行转换 (Pos -> Neg)
    if (n_pos_to_neg > 0) {
        std::vector<size_t> pos_indices;
        for (size_t i = 0; i < N; ++i) {
            if (types[i]) pos_indices.push_back(i);
        }
        std::shuffle(pos_indices.begin(), pos_indices.end(), gen);
        for (int i = 0; i < n_pos_to_neg; ++i) {
            size_t idx = pos_indices[i];
            types[idx] = false;
            Ks[idx] = K1;
        }
    }

    // 4. 执行转换 (Neg -> Pos)
    if (n_neg_to_pos > 0) {
        std::vector<size_t> neg_indices;
        for (size_t i = 0; i < N; ++i) {
            if (!types[i]) neg_indices.push_back(i);
        }
        std::shuffle(neg_indices.begin(), neg_indices.end(), gen);
        for (int i = 0; i < n_neg_to_pos; ++i) {
            size_t idx = neg_indices[i];
            types[idx] = true;
            Ks[idx] = K2;
        }
    }
}

/**
 * @brief 运行单次微观模拟并返回长时间平均 R
 */
double run_single_simulation(
    int N, double K1, double K2, double p0, double gamma,
    double r1, double r2, double dt, double t_max, unsigned int seed)
{
    // 为这次模拟设置种子
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

    // --- 2. 模拟 ---
    int num_steps = static_cast<int>(t_max / dt);
    // 丢弃前75%的数据
    int transient_steps = static_cast<int>(num_steps * 0.75);
    
    double R_sum = 0.0;
    int R_count = 0;

    for (int i = 0; i < num_steps; ++i) {
        if (i > 0) {
            // 化学反应
            chemical_reaction_tau_leaping(types, Ks, K1, K2, r1, r2, dt);
        }

        // 仅在稳定期计算 R
        cplx Z = calculate_Z(thetas);
        if (i >= transient_steps) {
            R_sum += std::abs(Z);
            R_count++;
        }

        // Kuramoto动力学
        heun_step_thetas(thetas, omegas, Ks, dt);
    }

    return (R_count > 0) ? (R_sum / R_count) : 0.0;
}

/**
 * @brief 将结果网格保存到 CSV 文件
 */
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

    // 写入表头 (X轴, p_s 值)
    outfile << "rate_scale(Y)";
    for (double p_s : x_axis) {
        outfile << "," << p_s;
    }
    outfile << "\n";

    // 写入数据 (Y轴, rate_scale)
    for (size_t i = 0; i < y_axis.size(); ++i) {
        outfile << y_axis[i]; // 写入行标题
        for (size_t j = 0; j < x_axis.size(); ++j) {
            outfile << "," << grid[i][j]; // 写入数据
        }
        outfile << "\n";
    }

    outfile.close();
    std::cout << "数据已成功保存到 scan_results.csv" << std::endl;
}

// ========= 主函数 (执行扫描) =========

int main() {
    // --- 1. 定义固定的全局参数 ---
    const int N = 15000;
    const double K1 = -0.5;
    const double K2 = 1;
    const double gamma = 0.05;
    const double dt = 0.02;
    const double t_max = 50.0;
    const unsigned int seed_base = 12345;

    // --- 2. 定义扫描参数范围 ---
    const int P_STEPS = 25; // X轴 (p_s)
    const int R_STEPS = 25; // Y轴 (rate_scale)

    std::vector<double> p_s_values = linspace(0.3, 0.95, P_STEPS);
    std::vector<double> rate_scale_values = linspace(0.001, 1.2, R_STEPS);

    // --- 3. 初始化结果存储网格 ---
    // R_steady_state[i][j] 对应 (rate_scale_values[i], p_s_values[j])
    std::vector<std::vector<double>> R_steady_state(R_STEPS, std::vector<double>(P_STEPS));

    std::cout << "--- 开始参数扫描 ---" << std::endl;
    std::cout << "参数: N=" << N << ", K1=" << K1 << ", K2=" << K2 << ", t_max=" << t_max << std::endl;
    std::cout << "扫描范围:" << std::endl;
    std::cout << "  X轴 (p_s): " << P_STEPS << " 步, 从 " << p_s_values.front() << " 到 " << p_s_values.back() << std::endl;
    std::cout << "  Y轴 (r1): " << R_STEPS << " 步, 从 " << rate_scale_values.front() << " 到 " << rate_scale_values.back() << std::endl;
    std::cout << "总计模拟次数: " << P_STEPS * R_STEPS << std::endl;

    // --- 4. 执行扫描循环 ---
    for (int i = 0; i < R_STEPS; ++i) {
        double rate_scale = rate_scale_values[i];
        
        // 显示Y轴进度
        std::cout << "\n处理 Y 步 " << (i + 1) << "/" << R_STEPS 
                  << " (rate_scale = " << rate_scale << ")" << std::endl;
        
        // C++ 中, 我们通常不使用 tqdm, 而是简单地打印点来显示X轴进度
        std::cout << "[" << std::flush;
        
        for (int j = 0; j < P_STEPS; ++j) {
            double p_s_val = p_s_values[j];

            // !!! 关键步骤: 设置本次模拟的参数 !!!
            double r1 = rate_scale;
            double p0 = p_s_val; // 从稳态比例开始
            
            double r2;
            
            // #################### 修正点 ####################
            // 原始错误: r2 = r1 * (1.0 - p_s_val) / p_s_val;
            // 正确公式 (p_s * r1 = (1-p_s) * r2):
            if (p_s_val < (1.0 - 1e-9)) {
                r2 = r1 * p_s_val / (1.0 - p_s_val);
            } else {
                // p_s 几乎为 1, r2 会趋于无穷大 (除非 r1=0)
                // 这种情况意味着 K1->K2 的速率极高
                // 为了数值稳定，设置一个极大值或根据情况处理
                // 但在这里，p_s=1 意味着 r1 必须为 0 才能稳态 (除非 r2=inf)
                // 稳妥起见，我们假设 p_s=1 意味着 r1=0。
                // 但在我们的扫描中, r1 是 rate_scale > 0。
                // 您的 p_s_values 不包含 1.0, 1e-9 的检查足够了。
                r2 = r1 * p_s_val / (1.0 - p_s_val);
            }

            if (p_s_val < 1e-9) { // 避免 p_s = 0 时 r2=0
                r2 = 0.0;
            }
            // #################################################
            

            // 为每次模拟设置唯一的、可复现的种子
            unsigned int sim_seed = seed_base + i * P_STEPS + j;

            // --- 运行单次模拟 ---
            double R_long_term = run_single_simulation(
                N, K1, K2, p0, gamma, r1, r2, dt, t_max, sim_seed
            );
            
            // 存储结果
            R_steady_state[i][j] = R_long_term;

            // 打印进度点
            std::cout << "." << std::flush;
        }
        std::cout << "]" << std::endl; // Y 步完成
    }

    std::cout << "\n--- 扫描完成 ---" << std::endl;

    // --- 5. 保存结果 ---
    save_results_to_csv(R_steady_state, p_s_values, rate_scale_values);

    return 0;
}