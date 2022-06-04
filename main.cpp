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

#ifdef _MSC_VER
inline unsigned int __builtin_clz(const unsigned int& x) {
    unsigned long r;
    _BitScanReverse(&r, x);
    return 31 - r;
}
inline unsigned long long __builtin_clzll(const unsigned long long& x) {
    unsigned long r;
    _BitScanReverse64(&r, x);
    return 63 - r;
}
#endif
#pragma warning(disable : 4146)
/*
iwi 先生の radix heap (https://github.com/iwiwi/radix-heap)
The MIT License (MIT)
Copyright (c) 2015 Takuya Akiba
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of the
Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
namespace radix_heap {
namespace internal {
template <bool Is64bit> class find_bucket_impl;

template <> class find_bucket_impl<false> {
  public:
    static inline constexpr size_t find_bucket(uint32_t x, uint32_t last) {
        return x == last ? 0 : 32 - __builtin_clz(x ^ last);
    }
};

template <> class find_bucket_impl<true> {
  public:
    static inline constexpr size_t find_bucket(uint64_t x, uint64_t last) {
        return x == last ? 0 : 64 - __builtin_clzll(x ^ last);
    }
};

template <typename T> inline constexpr size_t find_bucket(T x, T last) {
    return find_bucket_impl<sizeof(T) == 8>::find_bucket(x, last);
}

template <typename KeyType, bool IsSigned> class encoder_impl_integer;

template <typename KeyType> class encoder_impl_integer<KeyType, false> {
  public:
    typedef KeyType key_type;
    typedef KeyType unsigned_key_type;

    inline static constexpr unsigned_key_type encode(key_type x) { return x; }

    inline static constexpr key_type decode(unsigned_key_type x) { return x; }
};

template <typename KeyType> class encoder_impl_integer<KeyType, true> {
  public:
    typedef KeyType key_type;
    typedef typename std::make_unsigned<KeyType>::type unsigned_key_type;

    inline static constexpr unsigned_key_type encode(key_type x) {
        return static_cast<unsigned_key_type>(x) ^
               (unsigned_key_type(1) << unsigned_key_type(
                    std::numeric_limits<unsigned_key_type>::digits - 1));
    }

    inline static constexpr key_type decode(unsigned_key_type x) {
        return static_cast<key_type>(
            x ^ (unsigned_key_type(1)
                 << (std::numeric_limits<unsigned_key_type>::digits - 1)));
    }
};

template <typename KeyType, typename UnsignedKeyType>
class encoder_impl_decimal {
  public:
    typedef KeyType key_type;
    typedef UnsignedKeyType unsigned_key_type;

    inline static constexpr unsigned_key_type encode(key_type x) {
        return raw_cast<key_type, unsigned_key_type>(x) ^
               ((-(raw_cast<key_type, unsigned_key_type>(x) >>
                   (std::numeric_limits<unsigned_key_type>::digits - 1))) |
                (unsigned_key_type(1)
                 << (std::numeric_limits<unsigned_key_type>::digits - 1)));
    }

    inline static constexpr key_type decode(unsigned_key_type x) {
        return raw_cast<unsigned_key_type, key_type>(
            x ^
            (((x >> (std::numeric_limits<unsigned_key_type>::digits - 1)) - 1) |
             (unsigned_key_type(1)
              << (std::numeric_limits<unsigned_key_type>::digits - 1))));
    }

  private:
    template <typename T, typename U> union raw_cast {
      public:
        constexpr raw_cast(T t) : t_(t) {}
        operator U() const { return u_; }

      private:
        T t_;
        U u_;
    };
};

template <typename KeyType>
class encoder
    : public encoder_impl_integer<KeyType, std::is_signed<KeyType>::value> {};
template <>
class encoder<float> : public encoder_impl_decimal<float, uint32_t> {};
template <>
class encoder<double> : public encoder_impl_decimal<double, uint64_t> {};
} // namespace internal

template <typename KeyType, typename EncoderType = internal::encoder<KeyType>>
class radix_heap {
  public:
    typedef KeyType key_type;
    typedef EncoderType encoder_type;
    typedef typename encoder_type::unsigned_key_type unsigned_key_type;

    radix_heap() : size_(0), last_(), buckets_() {
        buckets_min_.fill(std::numeric_limits<unsigned_key_type>::max());
    }

    void push(key_type key) {
        const unsigned_key_type x = encoder_type::encode(key);
        assert(last_ <= x);
        ++size_;
        const size_t k = internal::find_bucket(x, last_);
        buckets_[k].emplace_back(x);
        buckets_min_[k] = std::min(buckets_min_[k], x);
    }

    key_type top() {
        pull();
        return encoder_type::decode(last_);
    }

    void pop() {
        pull();
        buckets_[0].pop_back();
        --size_;
    }

    size_t size() const { return size_; }

    bool empty() const { return size_ == 0; }

    void clear() {
        size_ = 0;
        last_ = key_type();
        for (auto& b : buckets_)
            b.clear();
        buckets_min_.fill(std::numeric_limits<unsigned_key_type>::max());
    }

    void swap(radix_heap<KeyType, EncoderType>& a) {
        std::swap(size_, a.size_);
        std::swap(last_, a.last_);
        buckets_.swap(a.buckets_);
        buckets_min_.swap(a.buckets_min_);
    }

  private:
    size_t size_;
    unsigned_key_type last_;
    std::array<std::vector<unsigned_key_type>,
               std::numeric_limits<unsigned_key_type>::digits + 1>
        buckets_;
    std::array<unsigned_key_type,
               std::numeric_limits<unsigned_key_type>::digits + 1>
        buckets_min_;

    void pull() {
        assert(size_ > 0);
        if (!buckets_[0].empty())
            return;

        size_t i;
        for (i = 1; buckets_[i].empty(); ++i)
            ;
        last_ = buckets_min_[i];

        for (unsigned_key_type x : buckets_[i]) {
            const size_t k = internal::find_bucket(x, last_);
            buckets_[k].emplace_back(x);
            buckets_min_[k] = std::min(buckets_min_[k], x);
        }
        buckets_[i].clear();
        buckets_min_[i] = std::numeric_limits<unsigned_key_type>::max();
    }
};

template <typename KeyType, typename ValueType,
          typename EncoderType = internal::encoder<KeyType>>
class pair_radix_heap {
  public:
    typedef KeyType key_type;
    typedef ValueType value_type;
    typedef EncoderType encoder_type;
    typedef typename encoder_type::unsigned_key_type unsigned_key_type;

    pair_radix_heap() : size_(0), last_(), buckets_() {
        buckets_min_.fill(std::numeric_limits<unsigned_key_type>::max());
    }

    void push(key_type key, const value_type& value) {
        const unsigned_key_type x = encoder_type::encode(key);
        assert(last_ <= x);
        ++size_;
        const size_t k = internal::find_bucket(x, last_);
        buckets_[k].emplace_back(x, value);
        buckets_min_[k] = std::min(buckets_min_[k], x);
    }

    void push(key_type key, value_type&& value) {
        const unsigned_key_type x = encoder_type::encode(key);
        assert(last_ <= x);
        ++size_;
        const size_t k = internal::find_bucket(x, last_);
        buckets_[k].emplace_back(x, std::move(value));
        buckets_min_[k] = std::min(buckets_min_[k], x);
    }

    template <class... Args> void emplace(key_type key, Args&&... args) {
        const unsigned_key_type x = encoder_type::encode(key);
        assert(last_ <= x);
        ++size_;
        const size_t k = internal::find_bucket(x, last_);
        buckets_[k].emplace_back(std::piecewise_construct,
                                 std::forward_as_tuple(x),
                                 std::forward_as_tuple(args...));
        buckets_min_[k] = std::min(buckets_min_[k], x);
    }

    key_type top_key() {
        pull();
        return encoder_type::decode(last_);
    }

    value_type& top_value() {
        pull();
        return buckets_[0].back().second;
    }

    void pop() {
        pull();
        buckets_[0].pop_back();
        --size_;
    }

    size_t size() const { return size_; }

    bool empty() const { return size_ == 0; }

    void clear() {
        size_ = 0;
        last_ = key_type();
        for (auto& b : buckets_)
            b.clear();
        buckets_min_.fill(std::numeric_limits<unsigned_key_type>::max());
    }

    void swap(pair_radix_heap<KeyType, ValueType, EncoderType>& a) {
        std::swap(size_, a.size_);
        std::swap(last_, a.last_);
        buckets_.swap(a.buckets_);
        buckets_min_.swap(a.buckets_min_);
    }

  private:
    size_t size_;
    unsigned_key_type last_;
    std::array<std::vector<std::pair<unsigned_key_type, value_type>>,
               std::numeric_limits<unsigned_key_type>::digits + 1>
        buckets_;
    std::array<unsigned_key_type,
               std::numeric_limits<unsigned_key_type>::digits + 1>
        buckets_min_;

    void pull() {
        assert(size_ > 0);
        if (!buckets_[0].empty())
            return;

        size_t i;
        for (i = 1; buckets_[i].empty(); ++i)
            ;
        last_ = buckets_min_[i];

        for (size_t j = 0; j < buckets_[i].size(); ++j) {
            const unsigned_key_type x = buckets_[i][j].first;
            const size_t k = internal::find_bucket(x, last_);
            buckets_[k].emplace_back(std::move(buckets_[i][j]));
            buckets_min_[k] = std::min(buckets_min_[k], x);
        }
        buckets_[i].clear();
        buckets_min_[i] = std::numeric_limits<unsigned_key_type>::max();
    }
};
} // namespace radix_heap

// =========================== ここまでライブラリ ===========================

struct Input {
    int N, T;
    Board<signed char, 10, 10> tiles;
    array<int, 16> stat;
    array<signed char, 17> tile_type_separations;

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
        auto cum_stat = 0;
        tile_type_separations[0] = 0;
        for (auto i = 0; i < 16; i++) {
            cum_stat += stat[i];
            tile_type_separations[i + 1] = cum_stat;
        }
    }
};

using Point = Vec2<int>;
constexpr auto kL = 1;
constexpr auto kU = 2;
constexpr auto kR = 4;
constexpr auto kD = 8;
auto input = Input();
static auto rng = Random(913418416u);

const auto kBoxDrawings =
    array<string, 16>{" ", "╸", "╹", "┛", "╺", "━", "┗", "┻",
                      "╻", "┓", "┃", "┫", "┏", "┳", "┣", "╋"};

// タイルの個数を数える
auto ComputeStat(const int n, const Board<signed char, 10, 10>& tiles) {
    auto stat = array<int, 16>();
    for (auto y = 0; y < n; y++) {
        for (auto x = 0; x < n; x++) {
            stat[tiles[{y, x}]]++;
        }
    }
    return stat;
}

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
    auto tiles = Board<signed char, 10, 10>();
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

void PrintTiles(const int n, const Board<signed char, 10, 10>& b) {
    for (auto y = 0; y < n; y++) {
        for (auto x = 0; x < n; x++) {
            cout << (int)b[{y, x}] << " ";
        }
        cout << endl;
    }
    for (auto y = 0; y < n; y++) {
        for (auto x = 0; x < n; x++) {
            cout << kBoxDrawings[b[{y, x}]];
        }
        cout << endl;
    }
}

// どこかの辺を切って、どこかの辺を足す
auto RandomTargetTree(const int n, const array<int, 16>& target_stat) {
    struct Edge {
        Point from, to;
    };
    static auto rng = Random(123478654);
    auto tiles = RandomSpaningTree(n);
    auto stat = array<int, 16>();
    for (auto y = 0; y < n; y++) {
        for (auto x = 0; x < n; x++) {
            stat[tiles[{y, x}]]++;
        }
    }
    for (auto trial = 0ll;; trial++) {
        if (trial % 1024 == 0) { // パラメータ
            tiles = RandomSpaningTree(n);
            stat = ComputeStat(n, tiles);
        }
        if ((trial & trial - 1) == 0) {
            // cout << "trial " << trial << endl;
            // PrintTiles(n, tiles);
            // for (auto c : stat)
            //     cout << c << " ";
            // cout << endl;
            // for (auto c : target_stat)
            //     cout << c << " ";
            // cout << endl;
        }

        {
            // 切る
            auto best = 1;
            auto best_edges = vector<Edge>();
            for (auto y = 0; y < n; y++) {
                for (auto x = 0; x < n; x++) {
                    if (x < n - 1 && tiles[{y, x}] & kR) {
                        auto gain = 0;

                        const auto l = Point{y, x};
                        const auto tile_l = tiles[l];
                        gain += stat[tile_l] > target_stat[tile_l];
                        const auto new_tile_l = tile_l ^ kR;
                        assert(new_tile_l < tile_l);
                        gain += stat[new_tile_l] < target_stat[new_tile_l];

                        const auto r = Point{y, x + 1};
                        const auto tile_r = tiles[r];
                        gain += stat[tile_r] > target_stat[tile_r];
                        const auto new_tile_r = tile_r ^ kL;
                        assert(new_tile_r < tile_r);
                        gain += stat[new_tile_r] < target_stat[new_tile_r];

                        if (gain == best) {
                            best_edges.push_back(Edge{l, r});
                        } else if (gain > best) {
                            best = gain;
                            best_edges.clear();
                            best_edges.push_back({l, r});
                        }
                    }
                    if (y < n - 1 && tiles[{y, x}] & kD) {
                        auto gain = 0;

                        const auto u = Point{y, x};
                        const auto tile_u = tiles[u];
                        gain += stat[tile_u] > target_stat[tile_u];
                        const auto new_tile_u = tile_u ^ kD;
                        assert(new_tile_u < tile_u);
                        gain += stat[new_tile_u] < target_stat[new_tile_u];

                        const auto d = Point{y + 1, x};
                        const auto tile_d = tiles[d];
                        gain += stat[tile_d] > target_stat[tile_d];
                        const auto new_tile_d = tile_d ^ kU;
                        assert(new_tile_d < tile_d);
                        gain += stat[new_tile_d] < target_stat[new_tile_d];

                        if (gain == best) {
                            best_edges.push_back({u, d});
                        } else if (gain > best) {
                            best = gain;
                            best_edges.clear();
                            best_edges.push_back({u, d});
                        }
                    }
                }
            }
            if (best_edges.size() == 0) {
                return tiles;
            }

            const auto rnd = rng.randint((int)best_edges.size());
            const auto edge = best_edges[rnd];
            if (edge.from.x < edge.to.x) {
                auto& tile_l = tiles[edge.from];
                auto& tile_r = tiles[edge.to];
                stat[tile_l]--;
                stat[tile_r]--;
                tile_l ^= kR;
                tile_r ^= kL;
                stat[tile_l]++;
                stat[tile_r]++;
            } else {
                auto& tile_u = tiles[edge.from];
                auto& tile_d = tiles[edge.to];
                stat[tile_u]--;
                stat[tile_d]--;
                tile_u ^= kD;
                tile_d ^= kU;
                stat[tile_u]++;
                stat[tile_d]++;
            }
        }
        {
            // 足す
            // DFS
            auto region = Board<signed char, 10, 10>();
            auto region_cnt = 1;
            for (auto y = 0; y < n; y++) {
                for (auto x = 0; x < n; x++) {
                    if (region[{y, x}])
                        continue;
                    region[{y, x}] = region_cnt;
                    auto stk = vector<Point>{Point{y, x}};
                    while (stk.size()) {
                        const auto v = stk.back();
                        stk.pop_back();
                        const auto tile = tiles[v];
                        if (tile & kD) {
                            const auto u = Point{v.y + 1, v.x};
                            if (region[u] == 0) {
                                region[u] = region_cnt;
                                stk.push_back(u);
                            }
                        }
                        if (tile & kR) {
                            const auto u = Point{v.y, v.x + 1};
                            if (region[u] == 0) {
                                region[u] = region_cnt;
                                stk.push_back(u);
                            }
                        }
                        if (tile & kU) {
                            const auto u = Point{v.y - 1, v.x};
                            if (region[u] == 0) {
                                region[u] = region_cnt;
                                stk.push_back(u);
                            }
                        }
                        if (tile & kL) {
                            const auto u = Point{v.y, v.x - 1};
                            if (region[u] == 0) {
                                region[u] = region_cnt;
                                stk.push_back(u);
                            }
                        }
                    }
                    region_cnt++;
                }
            }
            assert(region_cnt == 4);

            auto best = 0;
            auto best_edges = vector<Edge>();
            for (auto y = 0; y < n; y++) {
                for (auto x = 0; x < n; x++) {
                    if (x < n - 1 && region[{y, x}] != region[{y, x + 1}]) {
                        auto gain = 0;

                        const auto l = Point{y, x};
                        const auto tile_l = tiles[l];
                        gain += stat[tile_l] > target_stat[tile_l];
                        const auto new_tile_l = tile_l ^ kR;
                        assert(new_tile_l > tile_l);
                        gain += stat[new_tile_l] < target_stat[new_tile_l];

                        const auto r = Point{y, x + 1};
                        const auto tile_r = tiles[r];
                        gain += stat[tile_r] > target_stat[tile_r];
                        const auto new_tile_r = tile_r ^ kL;
                        assert(new_tile_r > tile_r);
                        gain += stat[new_tile_r] < target_stat[new_tile_r];

                        if (gain == best) {
                            best_edges.push_back(Edge{l, r});
                        } else if (gain > best) {
                            best = gain;
                            best_edges.clear();
                            best_edges.push_back({l, r});
                        }
                    }
                    if (y < n - 1 && region[{y, x}] != region[{y + 1, x}]) {
                        auto gain = 0;

                        const auto u = Point{y, x};
                        const auto tile_u = tiles[u];
                        gain += stat[tile_u] > target_stat[tile_u];
                        const auto new_tile_u = tile_u ^ kD;
                        assert(new_tile_u > tile_u);
                        gain += stat[new_tile_u] < target_stat[new_tile_u];

                        const auto d = Point{y + 1, x};
                        const auto tile_d = tiles[d];
                        gain += stat[tile_d] > target_stat[tile_d];
                        const auto new_tile_d = tile_d ^ kU;
                        assert(new_tile_d > tile_d);
                        gain += stat[new_tile_d] < target_stat[new_tile_d];

                        if (gain == best) {
                            best_edges.push_back({u, d});
                        } else if (gain > best) {
                            best = gain;
                            best_edges.clear();
                            best_edges.push_back({u, d});
                        }
                    }
                }
            }
            assert(best_edges.size() > 0);

            const auto rnd = rng.randint((int)best_edges.size());
            const auto edge = best_edges[rnd];
            if (edge.from.x < edge.to.x) {
                auto& tile_l = tiles[edge.from];
                auto& tile_r = tiles[edge.to];
                stat[tile_l]--;
                stat[tile_r]--;
                tile_l ^= kR;
                tile_r ^= kL;
                stat[tile_l]++;
                stat[tile_r]++;
            } else {
                auto& tile_u = tiles[edge.from];
                auto& tile_d = tiles[edge.to];
                stat[tile_u]--;
                stat[tile_d]--;
                tile_u ^= kD;
                tile_d ^= kU;
                stat[tile_u]++;
                stat[tile_d]++;
            }
        }
    }
}

auto NaiveAssignment(const int n, const Board<int, 50, 50>& A) {
    auto best = numeric_limits<int>::max();
    static auto permutation = array<int, 10>();
    iota(permutation.begin(), permutation.end(), 0);
    do {
        auto score = 0;
        for (auto y = 0; y < n; y++) {
            const auto x = permutation[y];
            score += A[{y + 1, x + 1}];
        }
        chmin(best, score);
    } while (next_permutation(permutation.begin(), permutation.begin() + n));
    return best;
}

// ハンガリアン法
// 参考: https://ei1333.github.io/luzhiled/snippets/graph/hungarian.html
auto Hungarian(const int n, const Board<int, 50, 50>& A) {
    // A は 1-based
    using T = int;
    const T infty = numeric_limits<T>::max();
    const int H = n + 1;
    const int W = n + 1;
    static auto P = array<int, 50>();
    fill(P.begin(), P.begin() + W, 0);
    static auto way = array<int, 50>();
    fill(way.begin(), way.begin() + W, 0);
    static auto U = array<int, 50>(); // 行 y の引いた数
    fill(U.begin(), U.begin() + H, 0);
    static auto V = array<int, 50>(); // 列 x の引いた数
    fill(V.begin(), V.begin() + W, 0);
    static auto minV = array<int, 50>(); // 列内の最小値
    static auto used = array<bool, 50>();

    for (int y = 1; y < H; y++) {
        P[0] = y;
        fill(minV.begin(), minV.begin() + W, infty);
        fill(used.begin(), used.begin() + W, false);
        int x0 = 0;
        while (P[x0] != 0) {
            const int y0 = P[x0];
            used[x0] = true;
            T delta = infty; // まだ処理してない部分全体の最小値
            int x1 = 0;      // delta の場所
            for (int x = 1; x < W; x++) {
                if (used[x])
                    continue;
                const T curr = A[{y0, x}] - U[y0] - V[x];
                if (curr < minV[x])
                    minV[x] = curr, way[x] = x0;
                if (minV[x] < delta)
                    delta = minV[x], x1 = x;
            }
            for (int x = 0; x < W; x++) {
                if (used[x])
                    U[P[x]] += delta, V[x] -= delta;
                else
                    minV[x] -= delta;
            }
            x0 = x1;
        }
        do {
            P[x0] = P[way[x0]];
            x0 = way[x0];
        } while (x0 != 0);
    }
    return -V[0];
}

void TestHungarian() {
    static auto rng = Random(1935831850);
    auto A = Board<int, 50, 50>();
    for (auto trial = 0; trial < 10000; trial++) {
        for (auto&& x : A.data) {
            x = rng.randint(20);
        }
        for (auto i = 1; i <= 9; i++) {
            auto t0 = Time();
            auto a1 = NaiveAssignment(i, A);
            auto t1 = Time();
            auto a2 = Hungarian(i, A);
            auto t2 = Time();
            // cout << "Naive:     " << a1 << " " << t1 - t0 << endl;
            // cout << "Hungarian: " << a2 << " " << t2 - t1 << endl;
            assert(a1 == a2);
        }
    }
}

auto TestSearchSpanningTree() {

    const auto t0 = Time();
    const auto b = RandomTargetTree(input.N, input.stat);
    const auto t1 = Time() - t0;
    PrintTiles(input.N, b);
    cout << "t1=" << t1 << endl;
    assert(ComputeStat(input.N, b) == input.stat);
}

using HashType = unsigned;
auto hash_table = array<array<array<HashType, 16>, 10>, 10>();

struct State {
    Board<signed char, 10, 10> tiles;
    HashType hash; // 衝突するとどうなる？？？
    short f;       // g + h
    short g;       // 現在までの距離
    short h;       // ヒューリスティック関数
    array<short, 16> hs; // ヒューリスティック関数 (タイルの種類ごと)
    array<unsigned char, 100>
        positions; // タイルの種類ごとの場所 y/x 4 ビットずつ
    int parent;
    static unsigned char temporal_move;

    inline void MoveTemporarily(const Point to) {
        // positions だけ変化させる
        const auto tile_type = tiles[to];
        const auto to_char = (unsigned char)(to.y << 4 | to.x);
        for (auto i = input.tile_type_separations[tile_type];
             i < input.tile_type_separations[tile_type + 1]; i++) {
            if (positions[i] == to_char) {
                temporal_move = i;
                swap(positions[0], positions[i]);
                return;
            }
        }
        assert(false);
    }

    inline void MoveBack() {
        swap(positions[0], positions[temporal_move]);
        temporal_move = 0xff;
    }

    void Print() const {
        cout << "State" << endl;
        PrintTiles(input.N, tiles);
        cout << "hash=" << hash << endl;
        cout << "f=" << f << endl;
        cout << "g=" << g << endl;
        cout << "h=" << h << endl;
        cout << "hs=";
        for (const auto hi : hs)
            cout << hi << ",";
        cout << endl;
        cout << "parent=" << parent << endl;
    }
};
unsigned char State::temporal_move;

auto state_buffer = Stack<State, 2000000>(); // 500 MB
constexpr auto sz_mb = sizeof(state_buffer) / 1024 / 1024;

struct StateAction {
    HashType hash; // 遷移先の状態のハッシュ
    int parent;
    short changed_h;
    unsigned char to; // 0 のタイルの移動先 y/x 4 ビットずつ

    inline auto G() const { return state_buffer[parent].g + 1; }
    inline auto H() const {
        const auto& state = state_buffer[parent];
        return state.h + changed_h -
               state.hs[state.tiles[{to >> 4, to & 0b1111}]];
    }
    inline auto F() const { return G() + H(); }

    inline auto ToState() const {
        const auto& old_state = state_buffer[parent];
        auto state = State();
        state.tiles = old_state.tiles;

        const auto tmp = state.tiles[{old_state.positions[0] >> 4,
                                      old_state.positions[0] & 0b1111}] == 0;
        assert(tmp);
        const auto tile_type = state.tiles[{to >> 4, to & 0b1111}];
        assert(tile_type != 0);
        state.tiles[{old_state.positions[0] >> 4,
                     old_state.positions[0] & 0b1111}] = tile_type;
        state.tiles[{to >> 4, to & 0b1111}] = 0;
        state.hash = hash;
        state.parent = parent;
        state.hs = old_state.hs;
        state.hs[tile_type] = changed_h;
        state.positions = old_state.positions;
        for (auto i = input.tile_type_separations[tile_type];
             i < input.tile_type_separations[tile_type + 1]; i++) {
            if (state.positions[i] == to) {
                swap(state.positions[0], state.positions[i]);
                goto break_ok;
            }
        }
        assert(false);
    break_ok:
        state.g = G();
        state.h = H();
        state.f = state.g + state.h;

        return state;
    }
};

template <class TilesType>
auto ComputeHash(const int n, const TilesType& tiles) {
    auto result = (HashType)0;
    for (auto y = 0; y < n; y++)
        for (auto x = 0; x < n; x++)
            result ^= hash_table[y][x][tiles[{y, x}]];
    return result;
}

template <class TilesType> auto ComputePositions(const TilesType& tiles) {
    auto result = array<unsigned char, 100>();
    auto indices = input.tile_type_separations;
    for (auto y = 0; y < input.N; y++)
        for (auto x = 0; x < input.N; x++)
            result[indices[tiles[{y, x}]]++] = (unsigned char)(y << 4 | x);
    return result;
}

// ヒューリスティック関数の計算
auto ComputeH(const State& state, const State& target_state,
              const int tile_type) {
    // tiles は使わず positions だけ見る
    assert(tile_type != 0);
    static auto cost_matrix = Board<int, 50, 50>();
    if (input.stat[tile_type] == 0)
        return 0;
    for (auto i = 0; i < input.stat[tile_type]; i++) {
        const auto yx =
            state.positions[input.tile_type_separations[tile_type] + i];
        const auto y = yx >> 4;
        const auto x = yx & 0b1111;
        for (auto j = 0; j < input.stat[tile_type]; j++) {
            const auto target_yx =
                target_state
                    .positions[input.tile_type_separations[tile_type] + j];
            const auto target_y = target_yx >> 4;
            const auto target_x = target_yx & 0b1111;
            cost_matrix[{i + 1, j + 1}] = abs(y - target_y) + abs(x - target_x);
        }
    }
    return Hungarian(input.stat[tile_type], cost_matrix);
}

// A* 探索
auto NaiveAStar() {
    auto distances = unordered_map<HashType, short>(); // 距離が確定
    auto q = radix_heap::pair_radix_heap<short, StateAction>(); //

    // 目標状態
    auto target_state = State();
    target_state.tiles = RandomTargetTree(input.N, input.stat);
    target_state.hash = ComputeHash(input.N, target_state.tiles);
    target_state.positions = ComputePositions(target_state.tiles);
    cout << "TARGET" << endl;
    target_state.Print();
    cout << endl;

    // 初期状態生成
    auto initial_state = State();
    initial_state.tiles = input.tiles;
    initial_state.hash = ComputeHash(input.N, initial_state.tiles);
    initial_state.positions = ComputePositions(initial_state.tiles);
    initial_state.g = 0;
    initial_state.h = 0;
    for (auto i = 1; i < 16; i++) {
        initial_state.h += initial_state.hs[i] =
            ComputeH(initial_state, target_state, i);
    }
    initial_state.f = initial_state.g + initial_state.h;
    initial_state.parent = -1;

    state_buffer.push(initial_state);
    distances[initial_state.hash] = initial_state.h;

    for (const auto d : {Point{0, 1}, {1, 0}, {0, -1}, {-1, 0}}) {
        auto action = StateAction();
        auto p_char = initial_state.positions[0];
        const auto v = Point{p_char >> 4, p_char & 0b1111};
        const auto u = v + d;
        if (!(0 <= u.x && u.x < input.N && 0 <= u.y && u.y < input.N))
            continue;
        action.parent = 0;
        action.to = (unsigned char)(u.y << 4 | u.x);
        const auto tile_type = initial_state.tiles[u];
        action.hash = initial_state.hash ^ hash_table[v.y][v.x][tile_type] ^
                      hash_table[u.y][u.x][tile_type];
        assert(initial_state.tiles[v] == 0);
        initial_state.MoveTemporarily(u);
        action.changed_h = ComputeH(initial_state, target_state, tile_type);
        initial_state.MoveBack();
        q.push(action.F(), action);
        distances[action.hash] = action.F();

        // sanity check
        {
            const auto s = action.ToState();
            for (auto i = 1; i < 16; i++) {
                const auto h_check = ComputeH(s, target_state, i);
                if (h_check != s.hs[i]) {
                    cout << "\nhが合わない、おしまい！！！" << endl;
                    s.Print();
                    assert(false);
                }
            }
        }
    }

    for (auto iteration = 0ll;; iteration++) {
        const auto fv = q.top_key();
        const auto old_action = q.top_value();
        q.pop();
        if (distances[old_action.hash] != fv)
            continue;

        const auto idx_state = state_buffer.size();
        state_buffer.push(old_action.ToState());
        auto& state = state_buffer.back();

        if ((iteration & iteration - 1) == 0) {
            cout << "iteration " << iteration << endl;
            state.Print();
        }

        for (const auto& d : {Point{0, 1}, {1, 0}, {0, -1}, {-1, 0}}) {
            auto action = StateAction();
            auto p_char = state.positions[0];
            const auto v = Point{p_char >> 4, p_char & 0b1111};
            const auto u = v + d;
            if (!(0 <= u.x && u.x < input.N && 0 <= u.y && u.y < input.N))
                continue;
            action.parent = idx_state;
            action.to = (unsigned char)(u.y << 4 | u.x);
            const auto tile_type = state.tiles[u];
            action.hash = state.hash ^ hash_table[v.y][v.x][tile_type] ^
                          hash_table[u.y][u.x][tile_type];
            auto distance_it = distances.find(action.hash);
            const auto found = distance_it != distances.end();
            if (found && distance_it->second <= state.f)
                continue;
            state.MoveTemporarily(u);
            action.changed_h = ComputeH(state, target_state, tile_type);
            state.MoveBack();
            const auto f = action.F();
            if (state.f > f) {
                // デバッグ
                cout << "やば！！！" << endl;
                state.Print();
                action.ToState().Print();
            }
            assert(state.f <= f);
            if (found && distance_it->second <= f)
                continue;
            if (found) {
                distance_it->second = f;
            } else {
                distances[action.hash] = f;
            }
            q.push(f, action);
            distances[action.hash] = f;

            if (action.hash == target_state.hash) {
                cout << "target state found!!" << endl;
                action.ToState().Print();
                return;
            }
        }
    }
}

auto Initialize() {
    input.Read();
    for (auto&& t : hash_table)
        for (auto&& tt : t)
            for (auto i = 1; i < 16; i++)
                tt[i] = rng.next();
}

int main() {
    Initialize();
    // TestSearchSpanningTree();
    // TestHungarian();
    NaiveAStar();
}
