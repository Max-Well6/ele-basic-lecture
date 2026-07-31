/*
 * hashtable.c —— 数据结构讲义配套示例（C 版本对照）
 *
 * 编译：gcc -std=c11 -Wall -O2 -o hashtable hashtable.c
 * 运行：./hashtable        （Windows 下为 hashtable.exe）
 *
 * 内容：
 *   1) 链地址法哈希表：BKDR 哈希 + 拉链解决冲突 + 负载因子超阈值自动扩容
 *   2) 开放地址法（线性探测）对照实现，演示"聚集"现象
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ==========================================================
 * 一、链地址法哈希表
 * ========================================================== */
typedef struct Entry {
    char *key;
    int value;
    struct Entry *next;   /* 同一桶内冲突元素串成链表 */
} Entry;

typedef struct {
    Entry **buckets;
    size_t capacity;
    size_t size;
} HashMap;

/* BKDR 哈希：种子 31/131/1313 都很常用，散列均匀且计算快 */
static unsigned long bkdr_hash(const char *str) {
    unsigned long seed = 131, hash = 0;
    while (*str) {
        hash = hash * seed + (unsigned char)(*str++);
    }
    return hash;
}

static HashMap *hm_create(size_t capacity) {
    HashMap *m = (HashMap *)malloc(sizeof(HashMap));
    m->capacity = capacity;
    m->size = 0;
    m->buckets = (Entry **)calloc(capacity, sizeof(Entry *));
    return m;
}

static void hm_put(HashMap *m, const char *key, int value);

/* 扩容：容量翻倍并 rehash 全部元素，均摊 O(1) */
static void hm_resize(HashMap *m) {
    size_t old_cap = m->capacity;
    Entry **old = m->buckets;

    m->capacity = old_cap * 2;
    m->buckets = (Entry **)calloc(m->capacity, sizeof(Entry *));
    m->size = 0;
    printf("  [扩容] %zu -> %zu 并 rehash\n", old_cap, m->capacity);

    for (size_t i = 0; i < old_cap; i++) {
        Entry *e = old[i];
        while (e) {
            Entry *next = e->next;
            hm_put(m, e->key, e->value);
            free(e->key);
            free(e);
            e = next;
        }
    }
    free(old);
}

static void hm_put(HashMap *m, const char *key, int value) {
    size_t idx = bkdr_hash(key) % m->capacity;
    for (Entry *e = m->buckets[idx]; e; e = e->next) {
        if (strcmp(e->key, key) == 0) {   /* 键已存在则更新 */
            e->value = value;
            return;
        }
    }
    Entry *node = (Entry *)malloc(sizeof(Entry));
    node->key = strdup(key);
    node->value = value;
    node->next = m->buckets[idx];         /* 头插入链 */
    m->buckets[idx] = node;
    m->size++;

    /* 负载因子 = size / capacity，超过 0.75 触发扩容 */
    if ((double)m->size / (double)m->capacity > 0.75) {
        hm_resize(m);
    }
}

static int hm_get(HashMap *m, const char *key, int *out) {
    size_t idx = bkdr_hash(key) % m->capacity;
    for (Entry *e = m->buckets[idx]; e; e = e->next) {
        if (strcmp(e->key, key) == 0) {
            *out = e->value;
            return 1;
        }
    }
    return 0;
}

static int hm_remove(HashMap *m, const char *key) {
    size_t idx = bkdr_hash(key) % m->capacity;
    Entry *prev = NULL, *e = m->buckets[idx];
    while (e) {
        if (strcmp(e->key, key) == 0) {
            if (prev) prev->next = e->next;
            else m->buckets[idx] = e->next;
            free(e->key);
            free(e);
            m->size--;
            return 1;
        }
        prev = e;
        e = e->next;
    }
    return 0;
}

static void hm_stats(HashMap *m) {
    size_t used = 0, max_chain = 0;
    for (size_t i = 0; i < m->capacity; i++) {
        size_t len = 0;
        for (Entry *e = m->buckets[i]; e; e = e->next) len++;
        if (len > 0) used++;
        if (len > max_chain) max_chain = len;
    }
    printf("  容量=%zu 元素=%zu 已用桶=%zu 负载因子=%.2f 最长链=%zu\n",
           m->capacity, m->size, used,
           (double)m->size / (double)m->capacity, max_chain);
}

static void hm_free(HashMap *m) {
    for (size_t i = 0; i < m->capacity; i++) {
        Entry *e = m->buckets[i];
        while (e) {
            Entry *next = e->next;
            free(e->key);
            free(e);
            e = next;
        }
    }
    free(m->buckets);
    free(m);
}

/* ==========================================================
 * 二、开放地址法（线性探测）对照
 * ========================================================== */
#define OA_CAP 13
#define OA_EMPTY  0
#define OA_FILLED 1

typedef struct {
    int state;
    int key;
} Slot;

static void oa_insert(Slot *table, int key) {
    size_t idx = (size_t)(key % OA_CAP);
    int probe = 0;
    while (table[idx].state == OA_FILLED) {
        if (table[idx].key == key) return;
        idx = (idx + 1) % OA_CAP;         /* 线性探测：往后找空位 */
        probe++;
    }
    table[idx].state = OA_FILLED;
    table[idx].key = key;
    printf("  插入 %3d -> 槽位 %2zu（探测 %d 次）\n", key, idx, probe);
}

/* ==========================================================
 * main
 * ========================================================== */
int main(void) {
    printf("===== 链地址法哈希表 =====\n");
    HashMap *m = hm_create(8);
    const char *keys[] = {"array", "list", "stack", "queue", "tree",
                          "graph", "heap", "hash", "trie", "bloom"};
    for (int i = 0; i < 10; i++) {
        hm_put(m, keys[i], i * 10);
    }
    hm_stats(m);

    int v;
    if (hm_get(m, "tree", &v)) printf("  get(\"tree\") 命中，值 = %d\n", v);
    printf("  get(\"none\") = %s\n", hm_get(m, "none", &v) ? "命中" : "未命中");

    hm_put(m, "tree", 999);
    hm_get(m, "tree", &v);
    printf("  更新后 tree = %d\n", v);

    printf("  remove(\"heap\") = %s\n", hm_remove(m, "heap") ? "成功" : "失败");
    hm_stats(m);
    hm_free(m);

    printf("\n===== 开放地址法（线性探测，容量 13）=====\n");
    Slot table[OA_CAP];
    memset(table, 0, sizeof(table));
    int nums[] = {12, 25, 38, 5, 18, 31, 7};
    for (int i = 0; i < 7; i++) {
        oa_insert(table, nums[i]);
    }
    printf("  最终表: ");
    for (int i = 0; i < OA_CAP; i++) {
        if (table[i].state == OA_FILLED) printf("[%d]=%d ", i, table[i].key);
    }
    printf("\n  注意 12/25/38 对 13 取模都等于 12，冲突后向后堆叠，产生连续聚集。\n");
    return 0;
}
