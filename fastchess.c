/*
 * fastchess.c - Fast C chess library with Python bindings.
 *
 * Implements mailbox-based board representation, legal move generation,
 * SAN parsing, board mirroring, and board-to-tensor conversion.
 * Designed as a high-performance drop-in for python-chess in the
 * Otter data pipeline.
 *
 * Build: python setup.py build_ext --inplace
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* ================================================================
 * Section 1: Constants and Data Structures
 * ================================================================ */

#define FC_WHITE  0
#define FC_BLACK  1

#define FC_PAWN   1
#define FC_KNIGHT 2
#define FC_BISHOP 3
#define FC_ROOK   4
#define FC_QUEEN  5
#define FC_KING   6

/* Piece encoding: type | (color << 3).
 * White pieces: 1-6, Black pieces: 9-14, Empty: 0. */
#define MAKE_PIECE(type, color) ((type) | ((color) << 3))
#define PIECE_TYPE(p) ((p) & 7)
#define PIECE_COLOR(p) ((p) >> 3)

#define SQ_RANK(sq) ((sq) >> 3)
#define SQ_FILE(sq) ((sq) & 7)
#define SQ(rank, file) (((rank) << 3) | (file))

/* Castling flags */
#define CASTLE_WK 1
#define CASTLE_WQ 2
#define CASTLE_BK 4
#define CASTLE_BQ 8

#define MAX_MOVES 256

typedef struct {
    int squares[64];     /* Mailbox representation */
    int turn;            /* FC_WHITE or FC_BLACK */
    int castling;        /* Bitmask of castling rights */
    int ep_square;       /* En passant target square, or -1 */
    int halfmove;        /* Halfmove clock */
    int fullmove;        /* Fullmove counter */
    int king_sq[2];      /* King squares: [WHITE], [BLACK] */
} FC_Board;

typedef struct {
    int from_sq;
    int to_sq;
    int promotion;       /* 0 = none, else piece type (KNIGHT..QUEEN) */
} FC_Move;

typedef struct {
    FC_Move moves[MAX_MOVES];
    int count;
} FC_MoveList;

/* Direction deltas for sliding pieces (rank_delta, file_delta) */
static const int ROOK_DR[4]   = { 1, -1,  0,  0};
static const int ROOK_DF[4]   = { 0,  0,  1, -1};
static const int BISHOP_DR[4] = { 1,  1, -1, -1};
static const int BISHOP_DF[4] = { 1, -1,  1, -1};

/* Knight deltas */
static const int KNIGHT_DR[8] = {-2, -2, -1, -1,  1,  1,  2,  2};
static const int KNIGHT_DF[8] = {-1,  1, -2,  2, -2,  2, -1,  1};

/* King deltas */
static const int KING_DR[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
static const int KING_DF[8] = {-1,  0,  1, -1, 1, -1, 0, 1};

/* ================================================================
 * Section 2: Board Initialization
 * ================================================================ */

static void fc_board_init(FC_Board *b) {
    memset(b, 0, sizeof(FC_Board));
    b->ep_square = -1;
    b->fullmove  = 1;
    b->turn      = FC_WHITE;
    b->castling  = CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ;

    static const int back[8] = {
        FC_ROOK, FC_KNIGHT, FC_BISHOP, FC_QUEEN,
        FC_KING, FC_BISHOP, FC_KNIGHT, FC_ROOK
    };
    for (int f = 0; f < 8; f++) {
        b->squares[SQ(0, f)] = MAKE_PIECE(back[f], FC_WHITE);
        b->squares[SQ(1, f)] = MAKE_PIECE(FC_PAWN, FC_WHITE);
        b->squares[SQ(6, f)] = MAKE_PIECE(FC_PAWN, FC_BLACK);
        b->squares[SQ(7, f)] = MAKE_PIECE(back[f], FC_BLACK);
    }
    b->king_sq[FC_WHITE] = SQ(0, 4); /* e1 */
    b->king_sq[FC_BLACK] = SQ(7, 4); /* e8 */
}

static int fc_board_from_fen(FC_Board *b, const char *fen) {
    memset(b, 0, sizeof(FC_Board));
    b->ep_square = -1;
    b->fullmove  = 1;

    /* 1. Piece placement */
    int rank = 7, file = 0;
    const char *p = fen;
    while (*p && *p != ' ') {
        if (*p == '/') {
            rank--;
            file = 0;
        } else if (*p >= '1' && *p <= '8') {
            file += (*p - '0');
        } else {
            int color = islower(*p) ? FC_BLACK : FC_WHITE;
            int pt = 0;
            switch (toupper(*p)) {
                case 'P': pt = FC_PAWN;   break;
                case 'N': pt = FC_KNIGHT; break;
                case 'B': pt = FC_BISHOP; break;
                case 'R': pt = FC_ROOK;   break;
                case 'Q': pt = FC_QUEEN;  break;
                case 'K': pt = FC_KING;   break;
                default: return 0;
            }
            int sq = SQ(rank, file);
            b->squares[sq] = MAKE_PIECE(pt, color);
            if (pt == FC_KING) b->king_sq[color] = sq;
            file++;
        }
        p++;
    }

    /* 2. Active color */
    if (*p == ' ') p++;
    if (*p == 'w')      b->turn = FC_WHITE;
    else if (*p == 'b') b->turn = FC_BLACK;
    else return 0;
    p++;

    /* 3. Castling */
    if (*p == ' ') p++;
    b->castling = 0;
    if (*p == '-') {
        p++;
    } else {
        while (*p && *p != ' ') {
            switch (*p) {
                case 'K': b->castling |= CASTLE_WK; break;
                case 'Q': b->castling |= CASTLE_WQ; break;
                case 'k': b->castling |= CASTLE_BK; break;
                case 'q': b->castling |= CASTLE_BQ; break;
            }
            p++;
        }
    }

    /* 4. En passant */
    if (*p == ' ') p++;
    if (*p == '-') {
        p++;
    } else if (*p >= 'a' && *p <= 'h') {
        int ef = *p - 'a'; p++;
        int er = *p - '1'; p++;
        b->ep_square = SQ(er, ef);
    }

    /* 5. Halfmove clock */
    if (*p == ' ') p++;
    b->halfmove = 0;
    while (*p >= '0' && *p <= '9') {
        b->halfmove = b->halfmove * 10 + (*p - '0');
        p++;
    }

    /* 6. Fullmove number */
    if (*p == ' ') p++;
    b->fullmove = 0;
    while (*p >= '0' && *p <= '9') {
        b->fullmove = b->fullmove * 10 + (*p - '0');
        p++;
    }
    if (b->fullmove == 0) b->fullmove = 1;

    return 1;
}

/* ================================================================
 * Section 3: Board Copy and Mirror
 * ================================================================ */

static void fc_board_copy(const FC_Board *src, FC_Board *dst) {
    memcpy(dst, src, sizeof(FC_Board));
}

static void fc_mirror_board(const FC_Board *src, FC_Board *dst) {
    memset(dst->squares, 0, sizeof(dst->squares));
    for (int sq = 0; sq < 64; sq++) {
        int piece = src->squares[sq];
        int msq = SQ(7 - SQ_RANK(sq), SQ_FILE(sq));
        if (piece == 0) {
            dst->squares[msq] = 0;
        } else {
            dst->squares[msq] = MAKE_PIECE(PIECE_TYPE(piece), 1 - PIECE_COLOR(piece));
        }
    }
    dst->turn = 1 - src->turn;

    /* Swap castling rights */
    dst->castling = 0;
    if (src->castling & CASTLE_WK) dst->castling |= CASTLE_BK;
    if (src->castling & CASTLE_WQ) dst->castling |= CASTLE_BQ;
    if (src->castling & CASTLE_BK) dst->castling |= CASTLE_WK;
    if (src->castling & CASTLE_BQ) dst->castling |= CASTLE_WQ;

    /* Mirror EP square - only keep if a capturing pawn exists in the
     * mirrored position (matches python-chess behavior). */
    dst->ep_square = -1;
    if (src->ep_square >= 0) {
        int mep = SQ(7 - SQ_RANK(src->ep_square), SQ_FILE(src->ep_square));
        /* In the mirrored board, dst->turn is already set.
         * Check if a pawn of dst->turn can capture on mep. */
        int cap_rank = (dst->turn == FC_WHITE) ? SQ_RANK(mep) - 1
                                               : SQ_RANK(mep) + 1;
        int mep_file = SQ_FILE(mep);
        int cap_pawn = MAKE_PIECE(FC_PAWN, dst->turn);
        int found = 0;
        if (cap_rank >= 0 && cap_rank < 8) {
            if (mep_file > 0 &&
                dst->squares[SQ(cap_rank, mep_file - 1)] == cap_pawn)
                found = 1;
            if (mep_file < 7 &&
                dst->squares[SQ(cap_rank, mep_file + 1)] == cap_pawn)
                found = 1;
        }
        if (found) dst->ep_square = mep;
    }

    /* Mirror king positions: white's new king = mirror of black's old king */
    dst->king_sq[FC_WHITE] = SQ(7 - SQ_RANK(src->king_sq[FC_BLACK]),
                                SQ_FILE(src->king_sq[FC_BLACK]));
    dst->king_sq[FC_BLACK] = SQ(7 - SQ_RANK(src->king_sq[FC_WHITE]),
                                SQ_FILE(src->king_sq[FC_WHITE]));

    dst->halfmove = src->halfmove;
    dst->fullmove = src->fullmove;
}

/* ================================================================
 * Section 4: Attack Detection
 * ================================================================ */

static int fc_is_attacked(const FC_Board *b, int sq, int by_color) {
    int r = SQ_RANK(sq), f = SQ_FILE(sq);

    /* Knight attacks */
    for (int i = 0; i < 8; i++) {
        int nr = r + KNIGHT_DR[i], nf = f + KNIGHT_DF[i];
        if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
            if (b->squares[SQ(nr, nf)] == MAKE_PIECE(FC_KNIGHT, by_color))
                return 1;
        }
    }

    /* King attacks */
    for (int i = 0; i < 8; i++) {
        int kr = r + KING_DR[i], kf = f + KING_DF[i];
        if (kr >= 0 && kr < 8 && kf >= 0 && kf < 8) {
            if (b->squares[SQ(kr, kf)] == MAKE_PIECE(FC_KING, by_color))
                return 1;
        }
    }

    /* Pawn attacks */
    if (by_color == FC_WHITE) {
        /* White pawns attack upward: a white pawn at (r-1, f±1) attacks sq */
        if (r > 0) {
            if (f > 0 && b->squares[SQ(r-1, f-1)] == MAKE_PIECE(FC_PAWN, FC_WHITE))
                return 1;
            if (f < 7 && b->squares[SQ(r-1, f+1)] == MAKE_PIECE(FC_PAWN, FC_WHITE))
                return 1;
        }
    } else {
        /* Black pawns attack downward: a black pawn at (r+1, f±1) attacks sq */
        if (r < 7) {
            if (f > 0 && b->squares[SQ(r+1, f-1)] == MAKE_PIECE(FC_PAWN, FC_BLACK))
                return 1;
            if (f < 7 && b->squares[SQ(r+1, f+1)] == MAKE_PIECE(FC_PAWN, FC_BLACK))
                return 1;
        }
    }

    /* Rook / Queen (straight rays) */
    for (int d = 0; d < 4; d++) {
        int cr = r + ROOK_DR[d], cf = f + ROOK_DF[d];
        while (cr >= 0 && cr < 8 && cf >= 0 && cf < 8) {
            int piece = b->squares[SQ(cr, cf)];
            if (piece != 0) {
                if (PIECE_COLOR(piece) == by_color) {
                    int pt = PIECE_TYPE(piece);
                    if (pt == FC_ROOK || pt == FC_QUEEN) return 1;
                }
                break;
            }
            cr += ROOK_DR[d];
            cf += ROOK_DF[d];
        }
    }

    /* Bishop / Queen (diagonal rays) */
    for (int d = 0; d < 4; d++) {
        int cr = r + BISHOP_DR[d], cf = f + BISHOP_DF[d];
        while (cr >= 0 && cr < 8 && cf >= 0 && cf < 8) {
            int piece = b->squares[SQ(cr, cf)];
            if (piece != 0) {
                if (PIECE_COLOR(piece) == by_color) {
                    int pt = PIECE_TYPE(piece);
                    if (pt == FC_BISHOP || pt == FC_QUEEN) return 1;
                }
                break;
            }
            cr += BISHOP_DR[d];
            cf += BISHOP_DF[d];
        }
    }

    return 0;
}

/* ================================================================
 * Section 5: Move Making
 * ================================================================ */

static void fc_make_move(FC_Board *b, const FC_Move *m) {
    int from = m->from_sq;
    int to   = m->to_sq;
    int piece = b->squares[from];
    int captured = b->squares[to]; /* May be 0 for non-captures */
    int pt    = PIECE_TYPE(piece);
    int color = b->turn;

    /* En passant capture: remove the pawn behind the EP square */
    if (pt == FC_PAWN && to == b->ep_square && b->ep_square >= 0) {
        int cap_sq = (color == FC_WHITE) ? to - 8 : to + 8;
        b->squares[cap_sq] = 0;
        captured = MAKE_PIECE(FC_PAWN, 1 - color); /* for halfmove reset */
    }

    /* Castling: move the rook */
    if (pt == FC_KING) {
        int base = (color == FC_WHITE) ? 0 : 56;
        if (from == base + 4 && to == base + 6) {
            /* Kingside */
            b->squares[base + 7] = 0;
            b->squares[base + 5] = MAKE_PIECE(FC_ROOK, color);
        } else if (from == base + 4 && to == base + 2) {
            /* Queenside */
            b->squares[base + 0] = 0;
            b->squares[base + 3] = MAKE_PIECE(FC_ROOK, color);
        }
    }

    /* Move piece */
    b->squares[to]   = piece;
    b->squares[from] = 0;

    /* Promotion */
    if (m->promotion != 0) {
        b->squares[to] = MAKE_PIECE(m->promotion, color);
    }

    /* Update king position */
    if (pt == FC_KING) {
        b->king_sq[color] = to;
    }

    /* Update castling rights */
    if (pt == FC_KING) {
        if (color == FC_WHITE) b->castling &= ~(CASTLE_WK | CASTLE_WQ);
        else                   b->castling &= ~(CASTLE_BK | CASTLE_BQ);
    }
    /* Rook moved or captured on corner squares */
    if (from == 0  || to == 0)  b->castling &= ~CASTLE_WQ; /* a1 */
    if (from == 7  || to == 7)  b->castling &= ~CASTLE_WK; /* h1 */
    if (from == 56 || to == 56) b->castling &= ~CASTLE_BQ; /* a8 */
    if (from == 63 || to == 63) b->castling &= ~CASTLE_BK; /* h8 */

    /* Update en passant square */
    if (pt == FC_PAWN && abs(to - from) == 16) {
        b->ep_square = (from + to) / 2;
    } else {
        b->ep_square = -1;
    }

    /* Halfmove clock */
    if (pt == FC_PAWN || captured != 0) {
        b->halfmove = 0;
    } else {
        b->halfmove++;
    }

    if (color == FC_BLACK) b->fullmove++;
    b->turn = 1 - color;
}

/* ================================================================
 * Section 6: Pseudo-Legal Move Generation
 * ================================================================ */

static void add_move(FC_MoveList *list, int from, int to, int promo) {
    if (list->count < MAX_MOVES) {
        FC_Move *m = &list->moves[list->count++];
        m->from_sq   = from;
        m->to_sq     = to;
        m->promotion = promo;
    }
}

static void gen_pawn_moves(const FC_Board *b, FC_MoveList *list) {
    int color = b->turn;
    int pawn  = MAKE_PIECE(FC_PAWN, color);
    int fwd   = (color == FC_WHITE) ? 1 : -1; /* rank direction */
    int start_rank = (color == FC_WHITE) ? 1 : 6;
    int promo_rank = (color == FC_WHITE) ? 7 : 0;

    for (int sq = 0; sq < 64; sq++) {
        if (b->squares[sq] != pawn) continue;
        int r = SQ_RANK(sq), f = SQ_FILE(sq);

        /* Single push */
        int nr = r + fwd;
        if (nr >= 0 && nr < 8) {
            int to = SQ(nr, f);
            if (b->squares[to] == 0) {
                if (nr == promo_rank) {
                    add_move(list, sq, to, FC_QUEEN);
                    add_move(list, sq, to, FC_ROOK);
                    add_move(list, sq, to, FC_BISHOP);
                    add_move(list, sq, to, FC_KNIGHT);
                } else {
                    add_move(list, sq, to, 0);
                }

                /* Double push */
                if (r == start_rank) {
                    int to2 = SQ(r + 2 * fwd, f);
                    if (b->squares[to2] == 0) {
                        add_move(list, sq, to2, 0);
                    }
                }
            }
        }

        /* Captures */
        int cap_df[2] = {-1, 1};
        for (int d = 0; d < 2; d++) {
            int cf = f + cap_df[d];
            if (cf < 0 || cf > 7) continue;
            int to = SQ(nr, cf);
            int target = b->squares[to];
            int is_capture = (target != 0 && PIECE_COLOR(target) != color);
            int is_ep = (to == b->ep_square && b->ep_square >= 0);

            if (is_capture || is_ep) {
                if (nr == promo_rank) {
                    add_move(list, sq, to, FC_QUEEN);
                    add_move(list, sq, to, FC_ROOK);
                    add_move(list, sq, to, FC_BISHOP);
                    add_move(list, sq, to, FC_KNIGHT);
                } else {
                    add_move(list, sq, to, 0);
                }
            }
        }
    }
}

static void gen_knight_moves(const FC_Board *b, FC_MoveList *list) {
    int color  = b->turn;
    int knight = MAKE_PIECE(FC_KNIGHT, color);

    for (int sq = 0; sq < 64; sq++) {
        if (b->squares[sq] != knight) continue;
        int r = SQ_RANK(sq), f = SQ_FILE(sq);

        for (int i = 0; i < 8; i++) {
            int nr = r + KNIGHT_DR[i], nf = f + KNIGHT_DF[i];
            if (nr < 0 || nr > 7 || nf < 0 || nf > 7) continue;
            int to = SQ(nr, nf);
            int target = b->squares[to];
            if (target == 0 || PIECE_COLOR(target) != color) {
                add_move(list, sq, to, 0);
            }
        }
    }
}

static void gen_sliding_moves(const FC_Board *b, FC_MoveList *list,
                              int piece_type,
                              const int *dr, const int *df, int ndirs) {
    int color = b->turn;
    int piece = MAKE_PIECE(piece_type, color);

    for (int sq = 0; sq < 64; sq++) {
        if (b->squares[sq] != piece) continue;
        int r = SQ_RANK(sq), f = SQ_FILE(sq);

        for (int d = 0; d < ndirs; d++) {
            int cr = r + dr[d], cf = f + df[d];
            while (cr >= 0 && cr < 8 && cf >= 0 && cf < 8) {
                int to = SQ(cr, cf);
                int target = b->squares[to];
                if (target == 0) {
                    add_move(list, sq, to, 0);
                } else {
                    if (PIECE_COLOR(target) != color)
                        add_move(list, sq, to, 0);
                    break;
                }
                cr += dr[d];
                cf += df[d];
            }
        }
    }
}

static void gen_king_moves(const FC_Board *b, FC_MoveList *list) {
    int color = b->turn;
    int king  = MAKE_PIECE(FC_KING, color);
    int sq    = b->king_sq[color];

    if (b->squares[sq] != king) return; /* Safety */

    int r = SQ_RANK(sq), f = SQ_FILE(sq);
    for (int i = 0; i < 8; i++) {
        int kr = r + KING_DR[i], kf = f + KING_DF[i];
        if (kr < 0 || kr > 7 || kf < 0 || kf > 7) continue;
        int to = SQ(kr, kf);
        int target = b->squares[to];
        if (target == 0 || PIECE_COLOR(target) != color) {
            add_move(list, sq, to, 0);
        }
    }
}

static void gen_castling_moves(const FC_Board *b, FC_MoveList *list) {
    int color = b->turn;
    int opp   = 1 - color;
    int base  = (color == FC_WHITE) ? 0 : 56;
    int king_sq = base + 4;

    if (b->king_sq[color] != king_sq) return;
    if (fc_is_attacked(b, king_sq, opp)) return;

    /* Kingside */
    int ks_flag = (color == FC_WHITE) ? CASTLE_WK : CASTLE_BK;
    if (b->castling & ks_flag) {
        if (b->squares[base + 5] == 0 &&
            b->squares[base + 6] == 0 &&
            b->squares[base + 7] == MAKE_PIECE(FC_ROOK, color) &&
            !fc_is_attacked(b, base + 5, opp) &&
            !fc_is_attacked(b, base + 6, opp)) {
            add_move(list, king_sq, base + 6, 0);
        }
    }

    /* Queenside */
    int qs_flag = (color == FC_WHITE) ? CASTLE_WQ : CASTLE_BQ;
    if (b->castling & qs_flag) {
        if (b->squares[base + 3] == 0 &&
            b->squares[base + 2] == 0 &&
            b->squares[base + 1] == 0 &&
            b->squares[base + 0] == MAKE_PIECE(FC_ROOK, color) &&
            !fc_is_attacked(b, base + 3, opp) &&
            !fc_is_attacked(b, base + 2, opp)) {
            add_move(list, king_sq, base + 2, 0);
        }
    }
}

static void fc_gen_pseudo_legal(const FC_Board *b, FC_MoveList *list) {
    list->count = 0;
    gen_pawn_moves(b, list);
    gen_knight_moves(b, list);
    gen_sliding_moves(b, list, FC_BISHOP, BISHOP_DR, BISHOP_DF, 4);
    gen_sliding_moves(b, list, FC_ROOK,   ROOK_DR,   ROOK_DF,   4);
    /* Queen: combine bishop + rook rays */
    gen_sliding_moves(b, list, FC_QUEEN,  BISHOP_DR, BISHOP_DF, 4);
    gen_sliding_moves(b, list, FC_QUEEN,  ROOK_DR,   ROOK_DF,   4);
    gen_king_moves(b, list);
    gen_castling_moves(b, list);
}

/* ================================================================
 * Section 7: Legal Move Generation
 * ================================================================ */

static void fc_gen_legal(const FC_Board *b, FC_MoveList *legal) {
    FC_MoveList pseudo;
    fc_gen_pseudo_legal(b, &pseudo);

    int moving_color = b->turn;
    int opp = 1 - moving_color;
    legal->count = 0;

    for (int i = 0; i < pseudo.count; i++) {
        FC_Board copy;
        fc_board_copy(b, &copy);
        fc_make_move(&copy, &pseudo.moves[i]);
        /* After make_move, copy.turn has flipped.
         * Check if the MOVING side's king is attacked by the opponent. */
        if (!fc_is_attacked(&copy, copy.king_sq[moving_color], opp)) {
            legal->moves[legal->count++] = pseudo.moves[i];
        }
    }
}

/* ================================================================
 * Section 8: UCI Conversion
 * ================================================================ */

static void fc_move_to_uci(const FC_Move *m, char *uci) {
    uci[0] = 'a' + SQ_FILE(m->from_sq);
    uci[1] = '1' + SQ_RANK(m->from_sq);
    uci[2] = 'a' + SQ_FILE(m->to_sq);
    uci[3] = '1' + SQ_RANK(m->to_sq);
    if (m->promotion) {
        static const char pc[] = " pnbrqk";
        uci[4] = pc[m->promotion];
        uci[5] = '\0';
    } else {
        uci[4] = '\0';
    }
}

static int fc_parse_uci(const char *uci, FC_Move *m) {
    int len = (int)strlen(uci);
    if (len < 4) return 0;

    int ff = uci[0] - 'a', fr = uci[1] - '1';
    int tf = uci[2] - 'a', tr = uci[3] - '1';
    if (ff < 0 || ff > 7 || fr < 0 || fr > 7) return 0;
    if (tf < 0 || tf > 7 || tr < 0 || tr > 7) return 0;

    m->from_sq = SQ(fr, ff);
    m->to_sq   = SQ(tr, tf);
    m->promotion = 0;

    if (len >= 5) {
        switch (uci[4]) {
            case 'q': m->promotion = FC_QUEEN;  break;
            case 'r': m->promotion = FC_ROOK;   break;
            case 'b': m->promotion = FC_BISHOP; break;
            case 'n': m->promotion = FC_KNIGHT; break;
        }
    }
    return 1;
}

/* ================================================================
 * Section 9: SAN Parsing
 * ================================================================ */

static int fc_parse_san(const FC_Board *b, const char *san, FC_Move *result) {
    /* Work on a cleaned copy (strip +, #, !, ?) */
    char clean[16];
    int len = 0;
    for (int i = 0; san[i] && i < 15; i++) {
        char c = san[i];
        if (c != '+' && c != '#' && c != '!' && c != '?')
            clean[len++] = c;
    }
    clean[len] = '\0';

    /* Castling */
    if (strcmp(clean, "O-O") == 0 || strcmp(clean, "0-0") == 0) {
        int base = (b->turn == FC_WHITE) ? 0 : 56;
        result->from_sq = base + 4;
        result->to_sq   = base + 6;
        result->promotion = 0;
        return 1;
    }
    if (strcmp(clean, "O-O-O") == 0 || strcmp(clean, "0-0-0") == 0) {
        int base = (b->turn == FC_WHITE) ? 0 : 56;
        result->from_sq = base + 4;
        result->to_sq   = base + 2;
        result->promotion = 0;
        return 1;
    }

    int piece_type;
    int disambig_file = -1;
    int disambig_rank = -1;
    int promotion = 0;

    int pos = 0;

    /* Piece type */
    if (clean[0] >= 'A' && clean[0] <= 'Z') {
        switch (clean[0]) {
            case 'K': piece_type = FC_KING;   break;
            case 'Q': piece_type = FC_QUEEN;  break;
            case 'R': piece_type = FC_ROOK;   break;
            case 'B': piece_type = FC_BISHOP; break;
            case 'N': piece_type = FC_KNIGHT; break;
            default: return 0;
        }
        pos = 1;
    } else {
        piece_type = FC_PAWN;
    }

    /* Check for promotion suffix (e.g. "=Q") */
    int promo_pos = -1;
    for (int i = 0; i < len; i++) {
        if (clean[i] == '=') { promo_pos = i; break; }
    }
    if (promo_pos >= 0 && promo_pos + 1 < len) {
        switch (clean[promo_pos + 1]) {
            case 'Q': promotion = FC_QUEEN;  break;
            case 'R': promotion = FC_ROOK;   break;
            case 'B': promotion = FC_BISHOP; break;
            case 'N': promotion = FC_KNIGHT; break;
            default: return 0;
        }
        len = promo_pos; /* Truncate */
    }

    /* Target square: last two characters */
    if (len < pos + 2) return 0;
    int tgt_f = clean[len - 2] - 'a';
    int tgt_r = clean[len - 1] - '1';
    if (tgt_f < 0 || tgt_f > 7 || tgt_r < 0 || tgt_r > 7) return 0;
    int target_sq = SQ(tgt_r, tgt_f);

    /* Disambiguation: characters between piece letter and target, ignoring 'x' */
    for (int i = pos; i < len - 2; i++) {
        char c = clean[i];
        if (c == 'x') continue;
        if (c >= 'a' && c <= 'h') disambig_file = c - 'a';
        else if (c >= '1' && c <= '8') disambig_rank = c - '1';
    }

    /* Generate legal moves and find the match */
    FC_MoveList legal;
    fc_gen_legal(b, &legal);

    FC_Move match;
    int n_match = 0;

    for (int i = 0; i < legal.count; i++) {
        FC_Move *m = &legal.moves[i];
        if (m->to_sq != target_sq) continue;
        if (m->promotion != promotion) continue;

        int fp = b->squares[m->from_sq];
        if (PIECE_TYPE(fp) != piece_type) continue;

        if (disambig_file >= 0 && SQ_FILE(m->from_sq) != disambig_file) continue;
        if (disambig_rank >= 0 && SQ_RANK(m->from_sq) != disambig_rank) continue;

        match = *m;
        n_match++;
    }

    if (n_match == 1) {
        *result = match;
        return 1;
    }
    return 0; /* Ambiguous or no match */
}

/* ================================================================
 * Section 10: Utility Functions
 * ================================================================ */

static int fc_has_legal_en_passant(const FC_Board *b) {
    if (b->ep_square < 0) return 0;

    FC_MoveList legal;
    fc_gen_legal(b, &legal);

    for (int i = 0; i < legal.count; i++) {
        if (legal.moves[i].to_sq == b->ep_square &&
            PIECE_TYPE(b->squares[legal.moves[i].from_sq]) == FC_PAWN)
            return 1;
    }
    return 0;
}

static int fc_is_in_check(const FC_Board *b) {
    return fc_is_attacked(b, b->king_sq[b->turn], 1 - b->turn);
}

/* ================================================================
 * Section 11: Board to Tensor (18 x 8 x 8)
 * ================================================================ */

static void fc_board_to_tensor(const FC_Board *board, float *tensor,
                               int do_canonical) {
    FC_Board canonical;
    const FC_Board *src;
    int original_turn = board->turn;

    if (do_canonical && original_turn == FC_BLACK) {
        fc_mirror_board(board, &canonical);
        src = &canonical;
    } else {
        src = board;
    }

    memset(tensor, 0, 18 * 64 * sizeof(float));

    /* Channels 0-11: pieces */
    for (int sq = 0; sq < 64; sq++) {
        int piece = src->squares[sq];
        if (piece == 0) continue;
        int pt = PIECE_TYPE(piece);
        int col = PIECE_COLOR(piece);
        int channel = (pt - 1) + (col * 6);
        int row = 7 - SQ_RANK(sq);
        int file = SQ_FILE(sq);
        tensor[channel * 64 + row * 8 + file] = 1.0f;
    }

    /* Channels 12-15: castling rights (fill entire plane) */
    if (src->castling & CASTLE_WK)
        for (int i = 0; i < 64; i++) tensor[12 * 64 + i] = 1.0f;
    if (src->castling & CASTLE_WQ)
        for (int i = 0; i < 64; i++) tensor[13 * 64 + i] = 1.0f;
    if (src->castling & CASTLE_BK)
        for (int i = 0; i < 64; i++) tensor[14 * 64 + i] = 1.0f;
    if (src->castling & CASTLE_BQ)
        for (int i = 0; i < 64; i++) tensor[15 * 64 + i] = 1.0f;

    /* Channel 16: en passant square (only if legal EP exists) */
    if (do_canonical) {
        /* Check on the canonical board */
        if (fc_has_legal_en_passant(src)) {
            int row = 7 - SQ_RANK(src->ep_square);
            int file = SQ_FILE(src->ep_square);
            tensor[16 * 64 + row * 8 + file] = 1.0f;
        }
    } else if (src->ep_square >= 0) {
        int row = 7 - SQ_RANK(src->ep_square);
        int file = SQ_FILE(src->ep_square);
        tensor[16 * 64 + row * 8 + file] = 1.0f;
    }

    /* Channel 17: original side to move (1.0 if white) */
    if (original_turn == FC_WHITE) {
        for (int i = 0; i < 64; i++) tensor[17 * 64 + i] = 1.0f;
    }
}

/* ================================================================
 * Section 12: FEN Generation (for debugging)
 * ================================================================ */

static void fc_board_to_fen(const FC_Board *b, char *fen, int maxlen) {
    static const char piece_chars[] = " PNBRQK  pnbrqk";
    char *p = fen;
    char *end = fen + maxlen - 10;

    for (int rank = 7; rank >= 0; rank--) {
        int empty = 0;
        for (int file = 0; file < 8; file++) {
            int piece = b->squares[SQ(rank, file)];
            if (piece == 0) {
                empty++;
            } else {
                if (empty > 0 && p < end) { *p++ = '0' + empty; empty = 0; }
                if (p < end) *p++ = piece_chars[piece];
            }
        }
        if (empty > 0 && p < end) *p++ = '0' + empty;
        if (rank > 0 && p < end) *p++ = '/';
    }

    if (p < end) *p++ = ' ';
    if (p < end) *p++ = (b->turn == FC_WHITE) ? 'w' : 'b';
    if (p < end) *p++ = ' ';

    if (b->castling == 0) {
        if (p < end) *p++ = '-';
    } else {
        if ((b->castling & CASTLE_WK) && p < end) *p++ = 'K';
        if ((b->castling & CASTLE_WQ) && p < end) *p++ = 'Q';
        if ((b->castling & CASTLE_BK) && p < end) *p++ = 'k';
        if ((b->castling & CASTLE_BQ) && p < end) *p++ = 'q';
    }

    if (p < end) *p++ = ' ';
    if (b->ep_square >= 0) {
        if (p < end) *p++ = 'a' + SQ_FILE(b->ep_square);
        if (p < end) *p++ = '1' + SQ_RANK(b->ep_square);
    } else {
        if (p < end) *p++ = '-';
    }

    if (p + 12 < fen + maxlen) {
        p += snprintf(p, end - p + 10, " %d %d", b->halfmove, b->fullmove);
    }
    *p = '\0';
}

/* ================================================================
 * Section 13: Python Board Type
 * ================================================================ */

typedef struct {
    PyObject_HEAD
    FC_Board board;
} BoardObject;

static PyTypeObject BoardType; /* Forward declaration */

/* Constructor (__init__) */
static int Board_init(BoardObject *self, PyObject *args, PyObject *kwds) {
    const char *fen = NULL;
    static char *kwlist[] = {"fen", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|s", kwlist, &fen))
        return -1;

    if (fen == NULL) {
        fc_board_init(&self->board);
    } else {
        if (!fc_board_from_fen(&self->board, fen)) {
            PyErr_SetString(PyExc_ValueError, "Invalid FEN string");
            return -1;
        }
    }
    return 0;
}

/* copy() -> Board */
static PyObject *Board_copy(BoardObject *self, PyObject *Py_UNUSED(args)) {
    BoardObject *new_board = (BoardObject *)BoardType.tp_alloc(&BoardType, 0);
    if (new_board == NULL) return NULL;
    fc_board_copy(&self->board, &new_board->board);
    return (PyObject *)new_board;
}

/* mirror() -> Board */
static PyObject *Board_mirror(BoardObject *self, PyObject *Py_UNUSED(args)) {
    BoardObject *new_board = (BoardObject *)BoardType.tp_alloc(&BoardType, 0);
    if (new_board == NULL) return NULL;
    fc_mirror_board(&self->board, &new_board->board);
    return (PyObject *)new_board;
}

/* push_san(san: str) -> str (UCI) */
static PyObject *Board_push_san(BoardObject *self, PyObject *args) {
    const char *san;
    if (!PyArg_ParseTuple(args, "s", &san)) return NULL;

    FC_Move move;
    if (!fc_parse_san(&self->board, san, &move)) {
        PyErr_Format(PyExc_ValueError, "Invalid or ambiguous SAN move: %s", san);
        return NULL;
    }

    char uci[7];
    fc_move_to_uci(&move, uci);
    fc_make_move(&self->board, &move);
    return PyUnicode_FromString(uci);
}

/* push_uci(uci: str) -> None */
static PyObject *Board_push_uci(BoardObject *self, PyObject *args) {
    const char *uci_str;
    if (!PyArg_ParseTuple(args, "s", &uci_str)) return NULL;

    FC_Move move;
    if (!fc_parse_uci(uci_str, &move)) {
        PyErr_Format(PyExc_ValueError, "Invalid UCI move: %s", uci_str);
        return NULL;
    }
    fc_make_move(&self->board, &move);
    Py_RETURN_NONE;
}

/* legal_moves_uci() -> list[str] */
static PyObject *Board_legal_moves_uci(BoardObject *self,
                                       PyObject *Py_UNUSED(args)) {
    FC_MoveList legal;
    fc_gen_legal(&self->board, &legal);

    PyObject *list = PyList_New(legal.count);
    if (!list) return NULL;

    for (int i = 0; i < legal.count; i++) {
        char uci[7];
        fc_move_to_uci(&legal.moves[i], uci);
        PyObject *s = PyUnicode_FromString(uci);
        if (!s) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, i, s);
    }
    return list;
}

/* piece_at(sq: int) -> (type, color) or None */
static PyObject *Board_piece_at(BoardObject *self, PyObject *args) {
    int sq;
    if (!PyArg_ParseTuple(args, "i", &sq)) return NULL;
    if (sq < 0 || sq > 63) {
        PyErr_SetString(PyExc_IndexError, "Square must be 0-63");
        return NULL;
    }

    int piece = self->board.squares[sq];
    if (piece == 0) Py_RETURN_NONE;

    return Py_BuildValue("(ii)", PIECE_TYPE(piece), PIECE_COLOR(piece));
}

/* is_check() -> bool */
static PyObject *Board_is_check(BoardObject *self, PyObject *Py_UNUSED(args)) {
    if (fc_is_in_check(&self->board))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/* has_kingside_castling_rights(color: int) -> bool */
static PyObject *Board_has_ks(BoardObject *self, PyObject *args) {
    int color;
    if (!PyArg_ParseTuple(args, "i", &color)) return NULL;
    int flag = (color == FC_WHITE) ? CASTLE_WK : CASTLE_BK;
    if (self->board.castling & flag) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/* has_queenside_castling_rights(color: int) -> bool */
static PyObject *Board_has_qs(BoardObject *self, PyObject *args) {
    int color;
    if (!PyArg_ParseTuple(args, "i", &color)) return NULL;
    int flag = (color == FC_WHITE) ? CASTLE_WQ : CASTLE_BQ;
    if (self->board.castling & flag) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/* has_legal_en_passant() -> bool */
static PyObject *Board_has_legal_ep(BoardObject *self,
                                    PyObject *Py_UNUSED(args)) {
    if (fc_has_legal_en_passant(&self->board)) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/* to_tensor(canonical=True) -> numpy.ndarray (18, 8, 8) */
static PyObject *Board_to_tensor(BoardObject *self, PyObject *args,
                                 PyObject *kwds) {
    int canonical = 1;
    static char *kwlist[] = {"canonical", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|p", kwlist, &canonical))
        return NULL;

    float data[18 * 8 * 8];
    fc_board_to_tensor(&self->board, data, canonical);

    npy_intp dims[3] = {18, 8, 8};
    PyObject *arr = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!arr) return NULL;
    memcpy(PyArray_DATA((PyArrayObject *)arr), data, sizeof(data));
    return arr;
}

/* fen() -> str */
static PyObject *Board_fen(BoardObject *self, PyObject *Py_UNUSED(args)) {
    char fen[128];
    fc_board_to_fen(&self->board, fen, sizeof(fen));
    return PyUnicode_FromString(fen);
}

/* __repr__ */
static PyObject *Board_repr(BoardObject *self) {
    char fen[128];
    fc_board_to_fen(&self->board, fen, sizeof(fen));
    return PyUnicode_FromFormat("Board('%s')", fen);
}

/* Properties */
static PyObject *Board_get_turn(BoardObject *self, void *closure) {
    (void)closure;
    return PyLong_FromLong(self->board.turn);
}

static PyObject *Board_get_ep(BoardObject *self, void *closure) {
    (void)closure;
    return PyLong_FromLong(self->board.ep_square);
}

static PyObject *Board_get_castling(BoardObject *self, void *closure) {
    (void)closure;
    return PyLong_FromLong(self->board.castling);
}

static PyObject *Board_get_halfmove(BoardObject *self, void *closure) {
    (void)closure;
    return PyLong_FromLong(self->board.halfmove);
}

static PyObject *Board_get_fullmove(BoardObject *self, void *closure) {
    (void)closure;
    return PyLong_FromLong(self->board.fullmove);
}

static PyMethodDef Board_methods[] = {
    {"copy",       (PyCFunction)Board_copy,       METH_NOARGS,
     "Return a copy of the board."},
    {"mirror",     (PyCFunction)Board_mirror,     METH_NOARGS,
     "Return a vertically mirrored copy with swapped colors."},
    {"push_san",   (PyCFunction)Board_push_san,   METH_VARARGS,
     "Parse and apply a SAN move. Returns the UCI string."},
    {"push_uci",   (PyCFunction)Board_push_uci,   METH_VARARGS,
     "Parse and apply a UCI move."},
    {"legal_moves_uci", (PyCFunction)Board_legal_moves_uci, METH_NOARGS,
     "Return list of all legal moves as UCI strings."},
    {"piece_at",   (PyCFunction)Board_piece_at,   METH_VARARGS,
     "Return (piece_type, color) at square, or None if empty."},
    {"is_check",   (PyCFunction)Board_is_check,   METH_NOARGS,
     "True if the side to move is in check."},
    {"has_kingside_castling_rights",
     (PyCFunction)Board_has_ks, METH_VARARGS,
     "True if color has kingside castling rights."},
    {"has_queenside_castling_rights",
     (PyCFunction)Board_has_qs, METH_VARARGS,
     "True if color has queenside castling rights."},
    {"has_legal_en_passant",
     (PyCFunction)Board_has_legal_ep, METH_NOARGS,
     "True if there is a legal en passant capture."},
    {"to_tensor",  (PyCFunction)Board_to_tensor,  METH_VARARGS | METH_KEYWORDS,
     "Return board as numpy array (18, 8, 8). canonical=True mirrors for black."},
    {"fen",        (PyCFunction)Board_fen,        METH_NOARGS,
     "Return the FEN string of the current position."},
    {NULL}
};

static PyGetSetDef Board_getset[] = {
    {"turn",     (getter)Board_get_turn,     NULL, "Side to move (0=WHITE, 1=BLACK)", NULL},
    {"ep_square",(getter)Board_get_ep,       NULL, "En passant target square (-1 if none)", NULL},
    {"castling", (getter)Board_get_castling, NULL, "Castling rights bitmask", NULL},
    {"halfmove", (getter)Board_get_halfmove, NULL, "Halfmove clock", NULL},
    {"fullmove", (getter)Board_get_fullmove, NULL, "Fullmove number", NULL},
    {NULL}
};

static PyTypeObject BoardType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "fastchess.Board",
    .tp_doc       = "Fast chess board representation.",
    .tp_basicsize = sizeof(BoardObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_new       = PyType_GenericNew,
    .tp_init      = (initproc)Board_init,
    .tp_repr      = (reprfunc)Board_repr,
    .tp_methods   = Board_methods,
    .tp_getset    = Board_getset,
};

/* ================================================================
 * Section 14: Module Definition
 * ================================================================ */

static PyModuleDef fastchess_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "fastchess",
    .m_doc  = "High-performance chess library for data pipelines.",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_fastchess(void) {
    import_array(); /* Initialize NumPy */

    if (PyType_Ready(&BoardType) < 0) return NULL;

    PyObject *m = PyModule_Create(&fastchess_module);
    if (m == NULL) return NULL;

    Py_INCREF(&BoardType);
    if (PyModule_AddObject(m, "Board", (PyObject *)&BoardType) < 0) {
        Py_DECREF(&BoardType);
        Py_DECREF(m);
        return NULL;
    }

    /* Constants */
    PyModule_AddIntConstant(m, "WHITE",  FC_WHITE);
    PyModule_AddIntConstant(m, "BLACK",  FC_BLACK);
    PyModule_AddIntConstant(m, "PAWN",   FC_PAWN);
    PyModule_AddIntConstant(m, "KNIGHT", FC_KNIGHT);
    PyModule_AddIntConstant(m, "BISHOP", FC_BISHOP);
    PyModule_AddIntConstant(m, "ROOK",   FC_ROOK);
    PyModule_AddIntConstant(m, "QUEEN",  FC_QUEEN);
    PyModule_AddIntConstant(m, "KING",   FC_KING);

    return m;
}
