#ifndef REVERSI_HPP
#define REVERSI_HPP

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <array>
#include <utility>
#include <vector>
#include <cassert>
using ULL = unsigned long long;
using std::array;
using std::vector;
using PII = std::pair<int, int>;

// 博弈树搜索最大深度
const int MAX_DEPTH = 6;
// 终局的最大深度
const int MAX_FINAL_DEPTH = 8;

// 无穷大
const int INF = 1e9;

// 棋盘大小
const int BOARD_WIDTH = 8;
const int BOARD_SIZE = 64;

// 哈希表大小
const int HASH_SIZE = 1 << 23;
const int HASH_MASK = HASH_SIZE - 1;

// 棋盘左上角序号为0
// 棋盘右下角序号为63
const int INDEX_N = -8;
const int INDEX_S = 8;
const int INDEX_W = -1;
const int INDEX_E = 1;
const int INDEX_NW = INDEX_N + INDEX_W;
const int INDEX_NE = INDEX_N + INDEX_E;
const int INDEX_SW = INDEX_S + INDEX_W;
const int INDEX_SE = INDEX_S + INDEX_E;

// 整个棋盘的移动
// 棋盘左上角在最低位
// 棋盘右下角在最高位
inline ULL ToN(ULL x) { return x >> 8u; }
inline ULL ToS(ULL x) { return x << 8u; }
inline ULL ToW(ULL x) { return ((x)&0xfefefefefefefefeull) >> 1u; }
inline ULL ToE(ULL x) { return ((x)&0x7f7f7f7f7f7f7f7full) << 1u; }
inline ULL ToNW(ULL x) { return ToN(ToW(x)); }
inline ULL ToNE(ULL x) { return ToN(ToE(x)); }
inline ULL ToSW(ULL x) { return ToS(ToW(x)); }
inline ULL ToSE(ULL x) { return ToS(ToE(x)); }
const array<ULL(*)(ULL), 8> all_dir = { ToN, ToS, ToW, ToE, ToNW, ToNE, ToSW, ToSE };
//ULL (*all_dir[8])(ULL) = { ToN, ToS, ToW, ToE, ToNW, ToNE, ToSW, ToSE };

// 计算 x 中比特1的数量
inline int CountBit(ULL x) { return __builtin_popcountll(x); }
// 只保留 x 最低位的比特
inline ULL LowBit(ULL x) { return x & (-x); }
// x 最低位比特的位置
inline int LowBitIndex(ULL x) { return __builtin_ffsll(x) - 1; }
// 二维下标转化成一维下标
inline int GetIndex(int x, int y)
{
    int idx = x * BOARD_WIDTH + y;
    return idx;
}
// 一维下标转化为二维下标
inline int GetX(int idx) { return idx / BOARD_WIDTH; }
inline int GetY(int idx) { return idx % BOARD_WIDTH; }
// 把 ULL 中为1的下标由小到大存入数组
vector<int> ULL2Vec(ULL x);
// 判断游戏处于何种时期
inline bool IsEarly(int step) { return step <= 25; }
inline bool IsMiddle(int step) { return step > 25 && step <= 45; }
inline bool IsLate(int step) { return step > 45; }

class Reversi
{
public:
    Reversi() = default;
    ~Reversi() {}
    // 初始化
    void Initialize();
    // 棋盘上第idx个位置是否为空
    inline bool IsEmpty(int idx) const { return (board_[0] >> idx & 1u) == 0 && (board_[1] >> idx & 1u) == 0; }
    // 得到所有空位置
    inline ULL GetEmpty() const { return ~(board_[0] | board_[1]); }
    // 切换玩家
    void ChangePlayer();
    // 下一次所有可以落子的位置，用ULL表示
    ULL GetAvailable() const;
    // 在棋盘上放一个棋子
    void PlacePiece(int idx);
    // 没有能落子的位置，跳过一回合
    // 每回合必须调用 PlacePiece 和 Skip 中的一个
    void Skip();
    // 在idx位置上落子之后，翻转周围的棋子
    void Flip(int idx);
    // 判断游戏是否结束
    bool IsOver() const;
    // 判断敌我棋子
    bool IsMyPiece(int x, int y) const;
    bool IsOppPiece(int x, int y) const;
    // 获得一方的棋盘
    ULL GetBoard(bool is_black) const;
    // 获得估计值
    int Evaluate() const;
    // 获得当前落子的玩家
    inline int GetPlayer() const { return player_; }
    // 获得哈希键值
    int GetHashKey() const;
    // 获得游戏步数
    inline int GetStep() const { return step_; }

    // 设置权重
    void SetWeights(unsigned char* w);
private:
    array<ULL, 2> board_;
    // player_==0 轮到黑方
    // player_==1 轮到白方
    // 黑方先下
    int player_;
    // 仅记录棋子权重方面的估计值
    int piece_eval_;
    // 连续跳过的次数，skip_==2游戏结束
    int skip_;
    // 步数
    int step_;
    // 哈希键值
    int key_;

    unsigned char weights[4] = {2, 255, 98, 16};
};


class AI
{
public:
    AI() {}
    ~AI() {}
    int Search(const Reversi &game) const;

private:
    PII AlphaBeta(const Reversi &game, int limit, int alpha, int beta) const;
};

enum class HashType { Alpha, Beta, Exact };
struct HashItem
{
    ULL black_board;
    ULL white_board;
    HashType type;
    int move;
    int eval;
    int limit;
};

void Insert(const Reversi &game, HashType type, int move, int eval, int limit);
PII Retrieve(const Reversi &game, int alpha, int beta, int limit, bool &prune);

#endif
