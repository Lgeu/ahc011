#include <atcoder/all>
#include <atcoder/maxflow.hpp>

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef __GNUC__
#include <x86intrin.h>
#endif

#ifdef __GNUC__
#pragma GCC target(                                                            \
    "sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native")
// #pragma GCC optimize("O3")
#pragma GCC optimize("Ofast")
//#pragma GCC optimize("unroll-loops")

#pragma clang attribute push(__attribute__((target("arch=skylake"))),          \
                             apply_to = function)
// 最後に↓を貼る
#ifdef __GNUC__
#pragma clang attribute pop
#endif
// 最後に↑を貼る
#endif

// ========================== macroes ==========================

//#define NDEBUG

#define rep(i, n) for (auto i = 0; (i) < (n); (i)++)
#define rep1(i, n) for (auto i = 1; (i) <= (n); (i)++)
#define rep3(i, s, n) for (auto i = (s); (i) < (n); (i)++)

#define ASSERT_RANGE(value, left, right)                                       \
    ASSERT((left <= value) && (value < right),                                 \
           "`%s` (%d) is out of range [%d, %d)", #value, value, left, right)

#define CHECK(var)                                                             \
    do {                                                                       \
        std::cout << #var << '=' << var << endl;                               \
    } while (false)

// ========================== utils ==========================

using namespace std;
using ll = long long;
constexpr double PI = 3.1415926535897932;

template <class T, class S> inline bool chmin(T& m, S q) {
    if (m > q) {
        m = q;
        return true;
    } else
        return false;
}

template <class T, class S> inline bool chmax(T& m, const S q) {
    if (m < q) {
        m = q;
        return true;
    } else
        return false;
}

// 乱数
struct Random {
    using ull = unsigned long long;
    unsigned seed;
    inline Random(const unsigned& seed_) : seed(seed_) { assert(seed != 0u); }
    const inline unsigned& next() {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return seed;
    }
    // (0.0, 1.0)
    inline double random() {
        return (double)next() * (1.0 / (double)0x100000000ull);
    }
    // [0, right)
    inline int randint(const int& right) { return (ull)next() * right >> 32; }
    // [left, right)
    inline int randint(const int& left, const int& right) {
        return ((ull)next() * (right - left) >> 32) + left;
    }
    template <class Container> inline auto choice(const Container& weight) {
        using T = typename Container::value_type;
        auto sum_weight = (T)0;
        for (const auto w : weight)
            sum_weight += w;
        if constexpr (is_floating_point_v<T>) {
            auto r = random() * sum_weight;
            auto cum_weight = (T)0;
            for (auto i = 0; i < (int)weight.size(); i++) {
                cum_weight += weight[i];
                if (r < cum_weight)
                    return i;
            }
            return (int)weight.size() - 1;
        } else {
            auto r = randint(sum_weight);
            auto cum_weight = (T)0;
            for (auto i = 0; i < (int)weight.size(); i++) {
                cum_weight += weight[i];
                if (r < cum_weight)
                    return i;
            }
            assert(false);
            return (int)weight.size() - 1;
        }
    }
};

// 2 次元ベクトル
template <typename T> struct Vec2 {
    /*
    y 軸正は下方向
    x 軸正は右方向
    回転は時計回りが正（y 軸正を上と考えると反時計回りになる）
    */
    T y, x;
    constexpr inline Vec2() = default;
    constexpr Vec2(const T& arg_y, const T& arg_x) : y(arg_y), x(arg_x) {}
    inline Vec2(const Vec2&) = default;            // コピー
    inline Vec2(Vec2&&) = default;                 // ムーブ
    inline Vec2& operator=(const Vec2&) = default; // 代入
    inline Vec2& operator=(Vec2&&) = default;      // ムーブ代入
    template <typename S>
    constexpr inline Vec2(const Vec2<S>& v) : y((T)v.y), x((T)v.x) {}
    inline Vec2 operator+(const Vec2& rhs) const {
        return Vec2(y + rhs.y, x + rhs.x);
    }
    inline Vec2 operator+(const T& rhs) const { return Vec2(y + rhs, x + rhs); }
    inline Vec2 operator-(const Vec2& rhs) const {
        return Vec2(y - rhs.y, x - rhs.x);
    }
    template <typename S> inline Vec2 operator*(const S& rhs) const {
        return Vec2(y * rhs, x * rhs);
    }
    inline Vec2 operator*(const Vec2& rhs) const { // x + yj とみなす
        return Vec2(x * rhs.y + y * rhs.x, x * rhs.x - y * rhs.y);
    }
    template <typename S> inline Vec2 operator/(const S& rhs) const {
        assert(rhs != 0.0);
        return Vec2(y / rhs, x / rhs);
    }
    inline Vec2 operator/(const Vec2& rhs) const { // x + yj とみなす
        return (*this) * rhs.inv();
    }
    inline Vec2& operator+=(const Vec2& rhs) {
        y += rhs.y;
        x += rhs.x;
        return *this;
    }
    inline Vec2& operator-=(const Vec2& rhs) {
        y -= rhs.y;
        x -= rhs.x;
        return *this;
    }
    template <typename S> inline Vec2& operator*=(const S& rhs) const {
        y *= rhs;
        x *= rhs;
        return *this;
    }
    inline Vec2& operator*=(const Vec2& rhs) { return *this = (*this) * rhs; }
    inline Vec2& operator/=(const Vec2& rhs) { return *this = (*this) / rhs; }
    inline bool operator!=(const Vec2& rhs) const {
        return x != rhs.x || y != rhs.y;
    }
    inline bool operator==(const Vec2& rhs) const {
        return x == rhs.x && y == rhs.y;
    }
    inline void rotate(const double& rad) { *this = rotated(rad); }
    inline Vec2<double> rotated(const double& rad) const {
        return (*this) * rotation(rad);
    }
    static inline Vec2<double> rotation(const double& rad) {
        return Vec2(sin(rad), cos(rad));
    }
    static inline Vec2<double> rotation_deg(const double& deg) {
        return rotation(PI * deg / 180.0);
    }
    inline Vec2<double> rounded() const {
        return Vec2<double>(round(y), round(x));
    }
    inline Vec2<double> inv() const { // x + yj とみなす
        const double norm_sq = l2_norm_square();
        assert(norm_sq != 0.0);
        return Vec2(-y / norm_sq, x / norm_sq);
    }
    inline double l2_norm() const { return sqrt(x * x + y * y); }
    inline double l2_norm_square() const { return x * x + y * y; }
    inline T l1_norm() const { return std::abs(x) + std::abs(y); }
    inline double abs() const { return l2_norm(); }
    inline double phase() const { // [-PI, PI) のはず
        return atan2(y, x);
    }
    inline double phase_deg() const { // [-180, 180) のはず
        return phase() * (180.0 / PI);
    }
    inline bool operator<(const Vec2& rhs) const {
        return y != rhs.y ? y < rhs.y : x < rhs.x;
    }
};
template <typename T, typename S>
inline Vec2<T> operator*(const S& lhs, const Vec2<T>& rhs) {
    return rhs * lhs;
}
template <typename T> ostream& operator<<(ostream& os, const Vec2<T>& vec) {
    os << vec.y << ' ' << vec.x;
    return os;
}

// 2 次元配列
template <class T, int height, int width> struct Board {
    array<T, height * width> data;
    template <class Int> constexpr inline auto& operator[](const Vec2<Int>& p) {
        return data[width * p.y + p.x];
    }
    template <class Int>
    constexpr inline const auto& operator[](const Vec2<Int>& p) const {
        return data[width * p.y + p.x];
    }
    template <class Int>
    constexpr inline auto& operator[](const initializer_list<Int>& p) {
        return data[width * *p.begin() + *(p.begin() + 1)];
    }
    template <class Int>
    constexpr inline const auto&
    operator[](const initializer_list<Int>& p) const {
        return data[width * *p.begin() + *(p.begin() + 1)];
    }
    constexpr inline void Fill(const T& fill_value) {
        fill(data.begin(), data.end(), fill_value);
    }
    void Print() const {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                cout << data[width * y + x] << " \n"[x == width - 1];
            }
        }
    }
};

// キュー
template <class T, int max_size> struct Queue {
    array<T, max_size> data;
    int left, right;
    inline Queue() : data(), left(0), right(0) {}
    inline Queue(initializer_list<T> init)
        : data(init.begin(), init.end()), left(0), right(init.size()) {}

    inline bool empty() const { return left == right; }
    inline void push(const T& value) {
        data[right] = value;
        right++;
    }
    inline void pop() { left++; }
    const inline T& front() const { return data[left]; }
    template <class... Args> inline void emplace(const Args&... args) {
        data[right] = T(args...);
        right++;
    }
    inline void clear() {
        left = 0;
        right = 0;
    }
    inline int size() const { return right - left; }
};

// スタック // コンストラクタ・デストラクタはちゃんと呼ぶ //
// 改修してからまともにテストしてないので注意
template <class T, int max_size> struct Stack {
    using value_type = T;
    array<char, max_size * sizeof(T)>
        buffer; // コンストラクタが呼ばれることを防ぐために char 型
    int right;
    inline Stack() : right(0) {} // buffer は初期化しない
    inline Stack(const int& n) : right(0) { resize(n); } // O(n)
    inline Stack(const int& n, const T& val) : right(0) { resize(n, val); }
    inline Stack(const initializer_list<T>& init) : right(0) {
        for (const auto& value : init)
            push(value);
    }
    inline Stack(const Stack& rhs) : right(0) { // コピー
        for (; right < rhs.right; right++)
            new (&data()[right]) T(rhs.data()[right]);
    }
    template <class S> inline Stack(const Stack<S, max_size>& rhs) : right() {
        for (; right < rhs.right; right++)
            new (&data()[right]) T(rhs.data()[right]);
    }
    template <class Container> Stack& operator=(const Container& rhs) {
        assert(rhs.size() <= max_size);
        clear();
        for (; right < rhs.right; right++)
            data()[right] = rhs[right];
        return *this;
    }
    Stack& operator=(Stack&&) = default;
    inline array<T, max_size>& data() { return *(array<T, max_size>*)&buffer; }
    inline bool empty() const { return 0 == right; }
    inline void push(const T& value) {
        assert(right < max_size);
        new (&data()[right++]) T(value);
    }
    inline T pop() { // これだけ標準ライブラリと仕様が違う
        assert(right > 0);
        auto tmp = data()[--right];
        data()[right].~T();
        return tmp;
    }
    const inline T& top() const { return data()[right - 1]; }
    template <class... Args> inline void emplace(const Args&... args) {
        assert(right < max_size);
        new (&data()[right++]) T(args...);
    }
    inline void clear() {
        while (right > 0)
            data()[--right].~T();
    }
    inline void insert(const int& idx, const T& value) {
        assert(0 <= idx && idx <= right);
        assert(right < max_size);
        push(value); // コンストラクタを呼ぶ
        for (int i = right - 1; i > idx; i--) {
            data()[i] = data()[i - 1];
        }
        data()[idx] = value;
    }
    inline void del(const int& idx) { // erase
        assert(0 <= idx && idx < right);
        right--;
        for (int i = idx; i < right; i++)
            data()[i] = data()[i + 1];
        data()[right].~T();
    }
    inline int index(const T& value) const {
        for (int i = 0; i < right; i++)
            if (value == data()[i])
                return i;
        return -1;
    }
    inline void remove(const T& value) {
        int idx = index(value);
        assert(idx != -1);
        del(idx);
    }
    inline void resize(const int& sz) {
        assert(0 <= sz && sz <= max_size);
        while (right < sz)
            new (&data()[right++]) T();
        while (right > sz)
            data()[--right].~T();
    }
    inline void resize(const int& sz, const T& fill_value) {
        assert(0 <= sz && sz <= max_size);
        while (right < sz)
            new (&data()[right++]) T(fill_value);
        while (right > sz)
            data()[--right].~T();
    }
    inline int size() const { return right; }
    inline T& operator[](const int& n) {
        assert(0 <= n && n < right);
        return data()[n];
    }
    inline const T& operator[](const int n) const {
        assert(0 <= n && n < right);
        return data()[n];
    }
    inline T* begin() { return (T*)&buffer; }
    inline const T* begin() const { return (const T*)&buffer; }
    inline T* end() { return begin() + right; }
    inline const T* end() const { return begin() + right; }
    inline T& front() {
        assert(right > 0);
        return data()[0];
    }
    const inline T& front() const {
        assert(right > 0);
        return data()[0];
    }
    inline T& back() {
        assert(right > 0);
        return data()[right - 1];
    }
    const inline T& back() const {
        assert(right > 0);
        return data()[right - 1];
    }
    inline bool contains(const T& value) const {
        for (const auto& dat : *this) {
            if (value == dat)
                return true;
        }
        return false;
    }
    inline void sort() { std::sort(begin(), end()); }
    template <class Compare> inline void sort(const Compare& comp) {
        std::sort(begin(), end(), comp);
    }
    inline void deduplicate() {
        sort();
        resize(std::unique(begin(), end()) - begin());
    }
    inline void Print(ostream& os = cout, const string& sep = " ") {
        for (int i = 0; i < right; i++) {
            os << data()[i] << (i == right - 1 ? "" : sep);
        }
        os << endl;
    }
};

template <class T, int size = 0x100000, class KeyType = unsigned>
struct MinimumHashMap {
    // ハッシュの値が size 以下
    array<T, size> data;
    Stack<int, size> used;
    constexpr static KeyType mask = size - 1;
    inline MinimumHashMap() {
        static_assert((size & size - 1) == 0, "not pow of 2");
        memset(&data[0], (unsigned char)-1, sizeof(data));
    }
    inline T& operator[](const KeyType& key) {
        if (data[key] == (T)-1)
            used.push(key);
        return data[key];
    }
    inline void clear() {
        for (const auto& key : used)
            data[key] = (T)-1;
        used.right = 0;
    }
};

// 時間 (秒)
inline double Time() {
    return static_cast<double>(
               chrono::duration_cast<chrono::nanoseconds>(
                   chrono::steady_clock::now().time_since_epoch())
                   .count()) *
           1e-9;
}

// 2 分法
template <class Container, class T>
inline int SearchSorted(const Container& vec, const T& a) {
    return lower_bound(vec.begin(), vec.end(), a) - vec.begin();
}

// argsort
template <typename T, int n, typename result_type, bool reverse = false>
inline auto Argsort(const array<T, n>& vec) {
    array<result_type, n> res;
    iota(res.begin(), res.end(), 0);
    sort(res.begin(), res.end(),
         [&](const result_type& l, const result_type& r) {
             return reverse ? vec[l] > vec[r] : vec[l] < vec[r];
         });
    return res;
}

// popcount  // SSE 4.2 を使うべき
inline int Popcount(const unsigned int& x) {
#ifdef _MSC_VER
    return (int)__popcnt(x);
#else
    return __builtin_popcount(x);
#endif
}
inline int Popcount(const unsigned long long& x) {
#ifdef _MSC_VER
    return (int)__popcnt64(x);
#else
    return __builtin_popcountll(x);
#endif
}

// x >> n & 1 が 1 になる最小の n ( x==0 は未定義 )
inline int CountRightZero(const unsigned int& x) {
#ifdef _MSC_VER
    unsigned long r;
    _BitScanForward(&r, x);
    return (int)r;
#else
    return __builtin_ctz(x);
#endif
}
inline int CountRightZero(const unsigned long long& x) {
#ifdef _MSC_VER
    unsigned long r;
    _BitScanForward64(&r, x);
    return (int)r;
#else
    return __builtin_ctzll(x);
#endif
}

inline double MonotonicallyIncreasingFunction(const double& h,
                                              const double& x) {
    // 0 < h < 1
    // f(0) = 0, f(1) = 1, f(0.5) = h
    assert(h > 0.0 && h < 1.0);
    if (h == 0.5)
        return x;
    const double& a = (1.0 - 2.0 * h) / (h * h);
    return expm1(log1p(a) * x) / a;
}

inline double MonotonicFunction(const double& start, const double& end,
                                const double& h, const double& x) {
    // h: x = 0.5 での進捗率
    return MonotonicallyIncreasingFunction(h, x) * (end - start) + start;
}

template <typename T> struct Slice {
    T *left, *right;
    inline Slice(T* const& l, T* const& r) : left(l), right(r) {}
    inline T* begin() { return left; }
    inline const T* begin() const { return (const T*)left; }
    inline T* end() { return right; }
    inline const T* end() const { return (const T*)right; }
    inline int size() const { return distance(left, right); }
    inline T& operator[](const int& idx) { return left[idx]; }
    inline const T& operator[](const int& idx) const { return left[idx]; }
};

struct Edge {
    int from, to;
    bool operator<(const Edge& rhs) const {
        return make_pair(from, to) < make_pair(rhs.from, rhs.to);
    }
};

template <int max_n, int max_m> struct Graph {
    int n, m;
    Stack<int, max_m> edges;
    Stack<int, max_n + 1> lefts;

    Graph() = default;
    template <class Container>
    Graph(const int& n_, const Container& arg_edges) {
        n = n_;
        m = arg_edges.size();
        static Stack<Edge, max_m> edges_;
        edges_.clear();
        for (const auto& e : arg_edges) {
            edges_.push(e);
        }
        edges_.sort();
        edges.resize(m);
        lefts.resize(n + 1);
        for (int i = 0; i < m; i++) {
            edges[i] = edges_[i].to;
        }
        auto idx_edges = 0;
        for (int v = 0; v <= n; v++) {
            lefts[v] = idx_edges;
            while (idx_edges < m && edges_[idx_edges].from == v) {
                idx_edges++;
            }
        }
    }
    inline Slice<int> operator[](const int& v) {
        return Slice<int>(edges.begin() + lefts[v],
                          edges.begin() + lefts[v + 1]);
    }
    inline Slice<int> operator[](const int& v) const {
        return Slice<int>(edges.begin() + lefts[v],
                          edges.begin() + lefts[v + 1]);
    }
};

// =========================================== ここまでライブラリ
// ===========================================

using Point = Vec2<int>;
constexpr auto kL = 1;
constexpr auto kU = 2;
constexpr auto kR = 4;
constexpr auto kD = 8;

const auto kBoxDrawings =
    array<string, 16>{" ", "╸", "╹", "┛", "╺", "━", "┗", "┻",
                      "╻", "┓", "┃", "┫", "┏", "┳", "┣", "╋"};

// ランダムに全域木をつくる
auto RandomSpaningTree(const int n) {
    static auto rng = Random(123478654);
    struct Edge {
        Point from, to;
    };
    auto edges = vector<Edge>();
    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < n; j++) {
            if (i + 1 < n && !(i + 1 == n - 1 && j == n - 1))
                edges.push_back({{i, j}, {i + 1, j}});
            if (j + 1 < n && !(i == n - 1 && j + 1 == n - 1))
                edges.push_back({{i, j}, {i, j + 1}});
        }
    }
    auto uf = atcoder::dsu(n * n);
    auto tiles = Board<int, 10, 10>();
    auto n_added_edges = 0;
    for (auto idx_edges = 0; n_added_edges < n * n - 2; idx_edges++) {
        const auto r = rng.randint(idx_edges, (int)edges.size());
        swap(edges[idx_edges], edges[r]);
        const auto& edge = edges[idx_edges];
        // cout << edge.from << " " << edge.to << endl;
        if (!uf.same(edge.from.y * n + edge.from.x,
                     edge.to.y * n + edge.to.x)) {
            uf.merge(edge.from.y * n + edge.from.x, edge.to.y * n + edge.to.x);
            if (edge.from.y + 1 == edge.to.y) {
                tiles[edge.from] |= 8;
                tiles[edge.to] |= 2;
            } else {
                tiles[edge.from] |= 4;
                tiles[edge.to] |= 1;
            }
            n_added_edges++;
        }
    }
    return tiles;
}

// ランダムに全域木を作ったときのタイルの数をばーって
auto TileCountStats() {
    constexpr auto sample_size = 10;
    constexpr auto n = 6;
    auto stats = map<array<int, 16>, int>();
    for (auto i = 0; i < sample_size; i++) {
        const auto tiles = RandomSpaningTree(n);
        auto stat = array<int, 16>();
        for (auto y = 0; y < n; y++)
            for (auto x = 0; x < n; x++)
                stat[tiles[{y, x}]]++;
        stats[stat]++;
    }
    for (const auto& [stat, count] : stats) {
        cout << "[";
        for (const auto& s : stat)
            cout << s << ",";
        cout << "], " << count << endl;
    }
}

// 目標となる数の全域木を適当に探してみる
auto SearchSpanningTree(const int n, const array<int, 16>& stat) {
    auto sum_stat = 0;
    for (const auto s : stat)
        sum_stat += s;
    assert(n * n == sum_stat);
    static auto rng = Random(123478654);
    for (auto trial = 0ll;; trial++) {
        if ((trial & trial - 1) == 0) {
            cout << "trial " << trial << endl;
        }
        auto tiles = Board<int, 10, 10>();
        auto remaining = stat;
        auto uf = atcoder::dsu(n * n);
        for (auto y = 0; y < n; y++) {
            for (auto x = 0; x < n; x++) {
                auto tile = 0;
                // 左 (1) と上 (2) は既に確定している
                if (x != 0 && tiles[{y, x - 1}] & kR) {
                    tile |= kL;
                }
                if (y != 0 && tiles[{y - 1, x}] & kD) {
                    tile |= kU;
                }
                if (tile & kU)
                    uf.merge(y * n + x, (y - 1) * n + x);
                if (tile & kL)
                    uf.merge(y * n + x, y * n + x - 1);
                // 候補は右と下の選び方 4 通り
                // ただし、右上のタイルが下を選んでおり、かつ既に自身と右上が同じ集合にいる場合、
                // 閉路ができるので右は選べない
                auto candidates = vector<int>();
                auto candidate_remaining_counts = vector<int>();
                for (const auto r : {0, kR}) {
                    if (x == n - 1 && r)
                        continue;
                    for (const auto d : {0, kD}) {
                        if (y == n - 1 && d)
                            continue;
                        if (r && x != n - 1 && y != 0 &&
                            tiles[{y - 1, x + 1}] & kD &&
                            uf.same(y * n + x, (y - 1) * n + x + 1))
                            continue;
                        const auto t = tile | r | d;
                        const auto p = remaining[t];
                        if (p) {
                            candidates.push_back(t);
                            candidate_remaining_counts.push_back(p);
                        }
                    }
                }
                if (candidates.size() == 0)
                    goto retry;
                tile = candidates[rng.choice(candidate_remaining_counts)];
                tiles[{y, x}] = tile;
                remaining[tile]--;
            }
        }
        return tiles;
    retry:;
    }
}

auto FlowSearch(const int n, const array<int, 16>& stat) {
    auto sum_stat = 0;
    for (const auto s : stat)
        sum_stat += s;
    assert(n * n == sum_stat);
    static auto rng = Random(123478654);
    const auto check_flow = [n](const auto& tiles, const auto& remaining,
                                const auto current_y, const auto current_x) {
        // if (current_y == n - 1)
        //     cout << "last line!!" << current_x << endl;
        if (current_y == n - 1 && current_x == n - 1)
            return true;
        auto mf = atcoder::mf_graph<int>(2 + 16 + (n - 1) * 4);
        constexpr auto kSource = 0;
        constexpr auto kSink = 1;
        constexpr auto kTileTypeOffset = 2;
        constexpr auto kPositionOffset = 18;
        auto positions = vector<Point>();
        auto idx_positions = 0;
        auto added = set<Point>();
        // 右上
        if (current_x == n - 1) {
            // 何もしない
        } else if (current_x == n - 2) {
            const auto y = current_y;
            const auto x = current_x + 1;
            if (current_y == n - 1) {
                const auto tile = (tiles[{y - 1, x}] & kD ? kU : 0) |
                                  (tiles[{y, x - 1}] & kR ? kL : 0);
                assert(remaining[tile] <= 1);
                return remaining[tile] == 1;
            }
            for (const auto d : {0, kD})
                mf.add_edge(kPositionOffset + idx_positions,
                            kTileTypeOffset +
                                (d | (tiles[{y - 1, x}] & kD ? kU : 0) |
                                 (tiles[{y, x - 1}] & kR ? kL : 0)),
                            1);
            added.insert({y, x});
            positions.push_back({y, x});
            idx_positions++;
        } else {
            const auto y = current_y;
            {
                const auto x = current_x + 1;
                if (y == n - 1) {
                    for (const auto r : {0, kR})
                        mf.add_edge(kPositionOffset + idx_positions,
                                    kTileTypeOffset +
                                        (r | (tiles[{y - 1, x}] & kD ? kU : 0) |
                                         (tiles[{y, x - 1}] & kR ? kL : 0)),
                                    1);
                } else {
                    for (const auto r : {0, kR})
                        for (const auto d : {0, kD})
                            mf.add_edge(kPositionOffset + idx_positions,
                                        kTileTypeOffset +
                                            (r | d |
                                             (tiles[{y - 1, x}] & kD ? kU : 0) |
                                             (tiles[{y, x - 1}] & kR ? kL : 0)),
                                        1);
                }
                added.insert({y, x});
                positions.push_back({y, x});
                idx_positions++;
            }
            {
                const auto x = n - 1;
                if (y == n - 1) {
                    for (const auto l : {0, kL})
                        mf.add_edge(kPositionOffset + idx_positions,
                                    kTileTypeOffset +
                                        (l | (tiles[{y - 1, x}] & kD ? kU : 0)),
                                    1);
                } else {
                    for (const auto l : {0, kL})
                        for (const auto d : {0, kD})
                            mf.add_edge(
                                kPositionOffset + idx_positions,
                                kTileTypeOffset +
                                    (l | d | (tiles[{y - 1, x}] & kD ? kU : 0)),
                                1);
                }
                added.insert({y, x});
                positions.push_back({y, x});
                idx_positions++;
            }
            for (auto x = current_x + 2; x < n - 1; x++) {
                if (y < n - 1) {
                    for (const auto r : {0, kR})
                        for (const auto l : {0, kL})
                            for (const auto d : {0, kD})
                                mf.add_edge(
                                    kPositionOffset + idx_positions,
                                    kTileTypeOffset +
                                        (l | r | d |
                                         (tiles[{y - 1, x}] & kD ? kU : 0)),
                                    1);
                } else if (y == n - 1) {
                    for (const auto r : {0, kR})
                        for (const auto l : {0, kL})
                            mf.add_edge(
                                kPositionOffset + idx_positions,
                                kTileTypeOffset +
                                    (l | r | (tiles[{y - 1, x}] & kD ? kU : 0)),
                                1);
                }
                added.insert({y, x});
                positions.push_back({y, x});
                idx_positions++;
            }
        }
        // 左上
        for (auto x = 0; x <= current_x; x++) {
            const auto y = current_y + 1;
            if (y < n - 1) {
                if (x == 0) {
                    for (const auto r : {0, kR})
                        for (const auto d : {0, kD})
                            mf.add_edge(
                                kPositionOffset + idx_positions,
                                kTileTypeOffset +
                                    (r | d | (tiles[{y - 1, x}] & kD ? kU : 0)),
                                1);
                } else if (x == n - 1) {
                    for (const auto l : {0, kL})
                        for (const auto d : {0, kD})
                            mf.add_edge(
                                kPositionOffset + idx_positions,
                                kTileTypeOffset +
                                    (l | d | (tiles[{y - 1, x}] & kD ? kU : 0)),
                                1);
                } else {
                    for (const auto r : {0, kR})
                        for (const auto l : {0, kL})
                            for (const auto d : {0, kD})
                                mf.add_edge(
                                    kPositionOffset + idx_positions,
                                    kTileTypeOffset +
                                        (l | r | d |
                                         (tiles[{y - 1, x}] & kD ? kU : 0)),
                                    1);
                }
                added.insert({y, x});
                positions.push_back({y, x});
                idx_positions++;
            } else if (y == n - 1) {
                if (x == 0) {
                    for (const auto r : {0, kR})
                        mf.add_edge(kPositionOffset + idx_positions,
                                    kTileTypeOffset +
                                        (r | (tiles[{y - 1, x}] & kD ? kU : 0)),
                                    1);
                } else if (x == n - 1) {
                    for (const auto l : {0, kL})
                        mf.add_edge(kPositionOffset + idx_positions,
                                    kTileTypeOffset +
                                        (l | (tiles[{y - 1, x}] & kD ? kU : 0)),
                                    1);
                } else {
                    for (const auto r : {0, kR})
                        for (const auto l : {0, kL})
                            mf.add_edge(
                                kPositionOffset + idx_positions,
                                kTileTypeOffset +
                                    (l | r | (tiles[{y - 1, x}] & kD ? kU : 0)),
                                1);
                }
                added.insert({y, x});
                positions.push_back({y, x});
                idx_positions++;
            }
        }
        // 左
        for (auto y = current_y + 2; y < n - 1; y++) {
            const auto x = 0;
            for (const auto u : {0, kU})
                for (const auto d : {0, kD})
                    for (const auto r : {0, kR})
                        mf.add_edge(kPositionOffset + idx_positions,
                                    kTileTypeOffset + (u | d | r), 1);

            added.insert({y, x});
            positions.push_back({y, x});
            idx_positions++;
        }
        // 右
        for (auto y = current_x == n - 1 ? current_y + 2 : current_y + 1;
             y < n - 1; y++) {
            const auto x = n - 1;
            for (const auto u : {0, kU})
                for (const auto d : {0, kD})
                    for (const auto l : {0, kL})
                        mf.add_edge(kPositionOffset + idx_positions,
                                    kTileTypeOffset + (u | d | l), 1);

            added.insert({y, x});
            positions.push_back({y, x});
            idx_positions++;
        }
        // 下
        if (current_y == n - 1) {
            // 何もしない
        } else if (current_y == n - 2) {
            const auto y = n - 1;
            if (current_x != n - 1) {
                const auto x = n - 1;
                for (const auto u : {0, kU})
                    for (const auto l : {0, kL})
                        mf.add_edge(kPositionOffset + idx_positions,
                                    kTileTypeOffset + (u | l), 1);

                added.insert({y, x});
                positions.push_back({y, x});
                idx_positions++;
            }
            for (auto x = current_x + 1; x < n - 1; x++) {
                for (const auto u : {0, kU})
                    for (const auto l : {0, kL})
                        for (const auto r : {0, kL})
                            mf.add_edge(kPositionOffset + idx_positions,
                                        kTileTypeOffset + (u | l | r), 1);

                added.insert({y, x});
                positions.push_back({y, x});
                idx_positions++;
            }
        } else {
            const auto y = n - 1;
            {
                const auto x = 0;
                for (const auto u : {0, kU})
                    for (const auto r : {0, kR})
                        mf.add_edge(kPositionOffset + idx_positions,
                                    kTileTypeOffset + (u | r), 1);

                added.insert({y, x});
                positions.push_back({y, x});
                idx_positions++;
            }
            {
                const auto x = n - 1;
                for (const auto u : {0, kU})
                    for (const auto l : {0, kL})
                        mf.add_edge(kPositionOffset + idx_positions,
                                    kTileTypeOffset + (u | l), 1);

                added.insert({y, x});
                positions.push_back({y, x});
                idx_positions++;
            }
            for (auto x = 1; x < n - 1; x++) {
                for (const auto u : {0, kU})
                    for (const auto l : {0, kL})
                        for (const auto r : {0, kL})
                            mf.add_edge(kPositionOffset + idx_positions,
                                        kTileTypeOffset + (u | l | r), 1);

                added.insert({y, x});
                positions.push_back({y, x});
                idx_positions++;
            }
        }

        {
            // cout << "current_y,x=" << current_y << current_x << endl;
            // cout << "added:     ";
            // for (const auto& p : added)
            //     cout << "(" << p.y << p.x << ")";
            // cout << endl;
            // cout << "positions: ";
            // // sort(positions.begin(), positions.end());
            // for (const auto& p : positions)
            //     cout << "(" << p.y << p.x << ")";
            // cout << endl;
        }
        assert(positions.size() == added.size());

        for (auto i = 0; i < (int)positions.size(); i++) {
            mf.add_edge(kSource, kPositionOffset + i, 1);
        }
        for (auto i = 0; i < 16; i++) {
            mf.add_edge(kTileTypeOffset + i, kSink, remaining[i]);
        }

        return mf.flow(kSource, kSink) == (int)positions.size();
    };
    for (auto trial = 0ll;; trial++) {
        if ((trial & trial - 1) == 0) {
            cout << "trial " << trial << endl;
        }
        auto tiles = Board<int, 10, 10>();
        auto remaining = stat;
        auto uf = atcoder::dsu(n * n);
        for (auto y = 0; y < n; y++) {
            for (auto x = 0; x < n; x++) {
                auto tile = 0;
                // 左 (1) と上 (2) は既に確定している
                if (x != 0 && tiles[{y, x - 1}] & kR) {
                    tile |= kL;
                }
                if (y != 0 && tiles[{y - 1, x}] & kD) {
                    tile |= kU;
                }
                if (tile & kU)
                    uf.merge(y * n + x, (y - 1) * n + x);
                if (tile & kL)
                    uf.merge(y * n + x, y * n + x - 1);
                // 候補は右と下の選び方 4 通り
                // ただし、右上のタイルが下を選んでおり、かつ既に自身と右上が同じ集合にいる場合、
                // 閉路ができるので右は選べない
                auto candidates = vector<int>();
                auto candidate_remaining_counts = vector<int>();
                for (const auto r : {0, kR}) {
                    if (x == n - 1 && r)
                        continue;
                    for (const auto d : {0, kD}) {
                        if (y == n - 1 && d)
                            continue;
                        if (r && x != n - 1 && y != 0 &&
                            tiles[{y - 1, x + 1}] & kD &&
                            uf.same(y * n + x, (y - 1) * n + x + 1))
                            continue;
                        const auto t = tile | r | d;
                        const auto p = remaining[t];
                        if (p) {
                            candidates.push_back(t);
                            candidate_remaining_counts.push_back(p);
                        }
                    }
                }
                do {
                    if (candidates.size() == 0)
                        goto retry;
                    const auto choice_index =
                        rng.choice(candidate_remaining_counts);
                    tile = candidates[choice_index];
                    tiles[{y, x}] = tile;
                    remaining[tile]--;
                    if (y < n - 4)
                        break;
                    if (check_flow(tiles, remaining, y, x))
                        break;
                    remaining[tile]++;
                    swap(candidates[choice_index],
                         candidates[candidates.size() - 1]);
                    swap(candidate_remaining_counts[choice_index],
                         candidate_remaining_counts
                             [candidate_remaining_counts.size() - 1]);
                    candidates.pop_back();
                    candidate_remaining_counts.pop_back();
                } while (1);
            }
        }
        return tiles;
    retry:;
    }
}

auto SwapSearch(const int n, const Board<int, 10, 10>& initial_tiles) {
    auto tiles = initial_tiles;
    static auto rng = Random(123478654);

    for (auto trial = 0ll;; trial++) {
        if ((trial & trial - 1) == 0) {
            cout << "trial " << trial << endl;
            tiles.Print();
            for (auto y = 0; y < n; y++) {
                for (auto x = 0; x < n; x++) {
                    cout << kBoxDrawings[tiles[{y, x}]];
                }
                cout << endl;
            }
        }
        auto p1 = Point{rng.randint(n), rng.randint(n)};
        if (((tiles[p1] & kR) > 0) ==
                (p1.x == n - 1 ? false
                               : ((tiles[{p1.y, p1.x + 1}] & kL) > 0)) &&
            ((tiles[p1] & kL) > 0) ==
                (p1.x == 0 ? false : ((tiles[{p1.y, p1.x - 1}] & kR) > 0)) &&
            ((tiles[p1] & kD) > 0) ==
                (p1.y == n - 1 ? false
                               : ((tiles[{p1.y + 1, p1.x}] & kU) > 0)) &&
            ((tiles[p1] & kU) > 0) ==
                (p1.y == 0 ? false : ((tiles[{p1.y - 1, p1.x}] & kD) > 0))) {
            continue;
        }
        auto p2 = Point{rng.randint(n), rng.randint(n)};
        if (((tiles[p2] & kR) > 0) ==
                (p2.x == n - 1 ? false
                               : ((tiles[{p2.y, p2.x + 1}] & kL) > 0)) &&
            ((tiles[p2] & kL) > 0) ==
                (p2.x == 0 ? false : ((tiles[{p2.y, p2.x - 1}] & kR) > 0)) &&
            ((tiles[p2] & kD) > 0) ==
                (p2.y == n - 1 ? false
                               : ((tiles[{p2.y + 1, p2.x}] & kU) > 0)) &&
            ((tiles[p2] & kU) > 0) ==
                (p2.y == 0 ? false : ((tiles[{p2.y - 1, p2.x}] & kD) > 0))) {
            continue;
        }
        swap(tiles[p1], tiles[p2]);

        // チェック

        for (auto y = 0; y < n; y++)
            for (auto x = 0; x < n; x++) {
                if (x != n - 1) {
                    if (((tiles[{y, x}] & kR) > 0) !=
                        ((tiles[{y, x + 1}] & kL) > 0))
                        goto next_trial;
                }
                if (y != n - 1) {
                    if (((tiles[{y, x}] & kD) > 0) !=
                        ((tiles[{y + 1, x}] & kU) > 0))
                        goto next_trial;
                }
            }
        return tiles;
        // 連結性確認してなかった
    next_trial:;
    }
}

struct Input {
    int N, T;
    Board<int, 10, 10> tiles;
    array<int, 16> stat;
    void Read() {
        cin >> N >> T;
        string tmp;
        fill(stat.begin(), stat.end(), 0);
        for (auto y = 0; y < N; y++) {
            cin >> tmp;
            for (auto x = 0; x < N; x++) {
                tiles[{y, x}] =
                    tmp[x] <= '9' ? tmp[x] - '0' : tmp[x] - 'a' + 10;
                stat[tiles[{y, x}]]++;
            }
        }
    }
};

void PrintTiles(const int n, const Board<int, 10, 10>& b) {
    b.Print();
    for (auto y = 0; y < n; y++) {
        for (auto x = 0; x < n; x++) {
            cout << kBoxDrawings[b[{y, x}]];
        }
        cout << endl;
    }
}

auto TestSearchSpanningTree() {
    // const auto b = SearchSpanningTree(
    //     n, array<int, 16>{1, 3, 2, 6, 2, 1, 2, 1, 3, 2, 3, 0, 5, 1, 2, 2});
    // const auto b = FlowSearch(
    //     n, array<int, 16>{1, 3, 2, 6, 2, 1, 2, 1, 3, 2, 3, 0, 5, 1, 2, 2});

    auto input = Input();
    input.Read();

    {
        // const auto t0 = Time();
        // const auto b = SearchSpanningTree(input.N, input.stat);
        // const auto t1 = Time() - t0;
        // PrintTiles(input.N, b);
        // cout << "t1=" << t1 << endl;
    }

    {
        const auto t0 = Time();
        const auto b = FlowSearch(input.N, input.stat);
        const auto t1 = Time() - t0;
        PrintTiles(input.N, b);
        cout << "t1=" << t1 << endl;
    }

    // const auto b = SwapSearch(input.N, input.tiles);

    // const auto b = WiseSearch(
    //     n, array<int, 16>{1, 3, 2, 6, 2, 1, 2, 1, 3, 2, 3, 0, 5, 1, 2, 2});
}

int main() {
    // TileCountStats();
    TestSearchSpanningTree();
}
