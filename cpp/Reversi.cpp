#include "Reversi.hpp"

const array<int, BOARD_SIZE> early_weight = {
    20, -20, 3, -15, -15, 3, -20, 20,
    -20, -5, -10, -2, -2, -10, -5, -20,
    3, -10, 11, 6, 6, 11, -10, 3,
    -15, -2, 6, 1, 1, 6, -2, -15,
    -15, -2, 6, 1, 1, 6, -2, -15,
    3, -10, 11, 6, 6, 11, -10, 3,
    -20, -5, -10, -2, -2, -10, -5, -20,
    20, -20, 3, -15, -15, 3, -20, 20};

const array<int, BOARD_SIZE> late_weight = {
    30, -2, -15, -5, -5, -15, -2, 30,
    -2, 7, -12, -10, -10, -12, 7, -2,
    -15, -12, 3, -7, -7, 3, -12, -15,
    -5, -10, -7, 5, 5, -7, -10, -5,
    -5, -10, -7, 5, 5, -7, -10, -5,
    -15, -12, 3, -7, -7, 3, -12, -15,
    -2, 7, -12, -10, -10, -12, 7, -2,
    30, -2, -15, -5, -5, -15, -2, 30};

// Zobrist hashing
const int hash_key[BOARD_SIZE][2] = {
    514999163, 344433251, 153449652, 780153524,
    384841693, 693301894, 958376243, 165280868,
    985155085, 208565809, 478202139, 850655193,
    969152852, 768233476, 834315952, 159694358,
    508220160, 919727181, 961378617, 455488695,
    499007717, 705028668, 481762934, 782244728,
    368422283, 683918260, 117918713, 612107214,
    717375760, 767166242, 887744298, 481749537,
    557089753, 858481755, 439936096, 623195554,
    575891839, 111311628, 162237439, 300117990,
    523683534, 967141116, 242974649, 575760706,
    280837425, 745258276, 163444188, 180853787,
    726199784, 896174371, 432057343, 751515474,
    101472271, 148413563, 964201354, 226527000,
    630486873, 714522393, 130994961, 267533532,
    485027913, 895922186, 488559745, 125880494,
    208040947, 246781093, 664878278, 795805013,
    372662155, 981728748, 604169915, 787048973,
    312527695, 787673307, 816498068, 268495130,
    540469743, 698856198, 894024927, 813574034,
    516371169, 474790440, 633396832, 230079425,
    206560135, 301165492, 949172069, 690525876,
    472740128, 533541400, 218303932, 938078228,
    335554129, 172159665, 632979326, 729257251,
    743414712, 317168518, 402840756, 408351105,
    678989927, 935306935, 564535993, 379652489,
    451429566, 996703318, 533568393, 671983686,
    219281613, 190093619, 327500204, 101311951,
    675132390, 526191986, 177012536, 590812041,
    693293363, 814987323, 647199275, 279513454,
    630146952, 370187667, 289964494, 598817795,
    255239697, 415566266, 611383202, 687929123};

HashItem hash_table[HASH_SIZE];

vector<int> ULL2Vec(ULL x)
{
    vector<int> ret;
    while (x)
    {
        ret.push_back(LowBitIndex(x));
        x -= LowBit(x);
    }
    return ret;
}

void Insert(const Reversi &game, HashType type, int move, int eval, int limit)
{
    int key = game.GetHashKey();

    hash_table[key].black_board = game.GetBoard(true);
    hash_table[key].white_board = game.GetBoard(false);
    hash_table[key].type = type;
    hash_table[key].move = move;
    hash_table[key].eval = eval;
    hash_table[key].limit = limit;
}

PII Retrieve(const Reversi &game, int alpha, int beta, int limit, bool &prune)
{
    int key = game.GetHashKey();
    HashItem &bucket = hash_table[key];

    prune = false;
    // bucket.limit<limit已经把bucket中没有记录的情况考虑进去了
    // 因为初始化的值是-1
    if (bucket.black_board != game.GetBoard(true) ||
        bucket.white_board != game.GetBoard(false) ||
        bucket.limit < limit)
        return PII(bucket.move, bucket.eval);

    if (bucket.type == HashType::Exact)
    {
        prune = true;
        return PII(bucket.move, bucket.eval);
    }
    if (bucket.type == HashType::Alpha && alpha >= bucket.eval)
    {
        prune = true;
        return PII(bucket.move, alpha);
    }
    if (bucket.type == HashType::Beta && beta <= bucket.eval)
    {
        prune = true;
        return PII(bucket.move, beta);
    }

    return PII(bucket.move, bucket.eval);
}

void Reversi::Initialize()
{
    board_[0] = board_[1] = 0ull;
    player_ = 0;
    piece_eval_ = 0;
    skip_ = 0;
    step_ = 0;
    key_ = hash_key[GetIndex(3, 4)][0] ^
           hash_key[GetIndex(4, 3)][0] ^
           hash_key[GetIndex(3, 3)][1] ^
           hash_key[GetIndex(4, 4)][1];

    board_[0] |= (1ull << GetIndex(3, 4)) | (1ull << GetIndex(4, 3));
    board_[1] |= (1ull << GetIndex(3, 3)) | (1ull << GetIndex(4, 4));

    // 初始化哈希表
    memset(hash_table, -1, sizeof(hash_table));
}

void Reversi::ChangePlayer()
{
    player_ ^= 1;
    piece_eval_ = -piece_eval_;
}

ULL Reversi::GetAvailable() const
{
    ULL pos = 0ull, tmp = 0ull, emp = GetEmpty();
    ULL my = board_[player_], opp = board_[player_ ^ 1];
    for (int i = 0; i < all_dir.size(); i++)
    {
        tmp = all_dir[i](my) & opp;
        for (int j = 0; j < 5; j++)
            tmp |= all_dir[i](tmp) & opp;
        pos |= all_dir[i](tmp) & emp;
    }
    return pos;
}

void Reversi::PlacePiece(int idx)
{
    //assert(idx >= 0 && idx < BOARD_SIZE);
    assert(1ull << idx & (GetEmpty()));
    board_[player_] |= 1ull << idx;
    skip_ = 0;
    step_++;
    key_ ^= hash_key[idx][player_];
    Flip(idx);
    ChangePlayer();
}

void Reversi::Skip()
{
    skip_++;
    ChangePlayer();
}

void Reversi::Flip(int idx)
{
    ULL &my = board_[player_], &opp = board_[player_ ^ 1];
    ULL pos = 1ull << idx;

    if (IsEarly(step_))
        piece_eval_ += 2 * early_weight[idx];
    else
        piece_eval_ += 2 * late_weight[idx];

    for (int i = 0; i < all_dir.size(); i++)
    {
        ULL tmp = all_dir[i](pos), mask = 0ull;
        int delta_val = 0, h = 0;
        while (tmp & opp)
        {
            if (IsEarly(step_))
                delta_val += early_weight[LowBitIndex(tmp)];
            else
                delta_val += late_weight[LowBitIndex(tmp)];
            h ^= hash_key[idx][player_ ^ 1];
            h ^= hash_key[idx][player_];
            mask |= tmp;
            tmp = all_dir[i](tmp);
        }
        if (tmp & my)
        {
            my ^= mask;
            opp ^= mask;
            // 对方得分减少 delta_val 己方得分增加 delta_val
            piece_eval_ += 2 * delta_val;
            key_ ^= h;
        }
    }
}

bool Reversi::IsOver() const
{
    if (skip_ >= 2)
        return true;
    if (!GetEmpty())
        return true;
    return false;
}

bool Reversi::IsMyPiece(int x, int y) const
{
    int idx = GetIndex(x, y);
    ULL pos = 1ull << idx;
    return board_[player_] & pos;
}

bool Reversi::IsOppPiece(int x, int y) const
{
    int idx = GetIndex(x, y);
    ULL pos = 1ull << idx;
    return board_[player_ ^ 1] & pos;
}

ULL Reversi::GetBoard(bool is_black) const
{
    if (is_black)
        return board_[0];
    return board_[1];
}

int Reversi::GetHashKey() const
{
    // 最后一位标识玩家
    return (((key_ & HASH_MASK) << 1) | player_) & HASH_MASK;
}

int Reversi::Evaluate() const
{
//    // 棋子位置权值
//    const int w1 = 6;
//    // 稳定点权值
//    const int w2 = 5000;
//    // 行动力权值
//    const int w3 = 150;
//    // 棋子数目权值
//    const int w4 = 10;

    int piece = piece_eval_;

    int my_stable = 0, opp_stable = 0;
    if (IsMyPiece(0, 0))
        my_stable++;
    else if (IsOppPiece(0, 0))
        opp_stable++;
    if (IsMyPiece(0, 7))
        my_stable++;
    else if (IsOppPiece(0, 7))
        opp_stable++;
    if (IsMyPiece(7, 0))
        my_stable++;
    else if (IsOppPiece(7, 0))
        opp_stable++;
    if (IsMyPiece(7, 7))
        my_stable++;
    else if (IsOppPiece(7, 7))
        opp_stable++;
    int stable = my_stable - opp_stable;

    int mobility = CountBit(GetAvailable());

    int my_pieces = CountBit(GetBoard(player_ ^ 1));
    int opp_pieces = CountBit(GetBoard(player_));
    int piece_diff = my_pieces - opp_pieces;

    if (IsOver())
    {
        if (piece_diff > 0)
            return INF - 1;
        // 防止估值不在(alpha,beta)范围内
        return -INF + 1;
    }

    return weights[0] * piece + weights[1] * stable + weights[2] * mobility + weights[3] * piece_diff;
}

void Reversi::SetWeights(unsigned char* w) {
    for(int i = 0; i < 4; i++){
        weights[i] = w[i];
    }
}

int AI::Search(const Reversi &game) const
{
    if (IsLate(game.GetStep()))
        return AlphaBeta(game, MAX_FINAL_DEPTH, -INF, INF).first;
    return AlphaBeta(game, MAX_DEPTH, -INF, INF).first;
}

PII AI::AlphaBeta(const Reversi &game, int limit, int alpha, int beta) const
{
    if (game.IsOver() || limit == 0)
        return PII(-1, game.Evaluate());

    bool prune = false;
    PII last = Retrieve(game, alpha, beta, limit, prune);

    if (prune)
        return last;

    vector<int> next_moves = ULL2Vec(game.GetAvailable());
    for (int i = 0; i < next_moves.size(); i++)
    {
        if (next_moves[i] == last.first)
        {
            std::swap(next_moves[i], next_moves[0]);
            break;
        }
    }
    if (next_moves.empty())
    {
        Reversi child = game;
        child.Skip();
        int val = -AlphaBeta(child, limit - 1, -beta, -alpha).second;
        return PII(-1, val);
    }

    int best_move = -1;
    HashType type = HashType::Alpha;
    for (int i = 0; i < next_moves.size(); i++)
    {
        Reversi child = game;
        child.PlacePiece(next_moves[i]);
        int val = -AlphaBeta(child, limit - 1, -beta, -alpha).second;
        if (val >= beta)
        {
            best_move = next_moves[i];
            Insert(game, HashType::Beta, best_move, beta, limit);
            return PII(best_move, beta);
        }
        if (val > alpha)
        {
            best_move = next_moves[i];
            alpha = val;
            type = HashType::Exact;
        }
    }
    Insert(game, type, best_move, alpha, limit);

    return PII(best_move, alpha);
}
