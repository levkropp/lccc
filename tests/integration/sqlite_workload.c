// SQLite real-workload test for LCCC: multi-table transactions, joins,
// aggregates, indexes, correlated subqueries, UPDATE/DELETE with WHERE.
// Prints a deterministic transcript that is diffed against the GCC build.
#include <stdio.h>
#include <string.h>
#include "sqlite3.h"

static sqlite3 *db;
static char *err = NULL;

static void exec_ok(const char *sql) {
    int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
    if (rc != SQLITE_OK) {
        printf("FAIL: %s: %s\n", sql, err ? err : "?");
        sqlite3_free(err);
        err = NULL;
    }
}

static int print_row(void *ud, int argc, char **argv, char **cols) {
    (void)ud; (void)cols;
    for (int i = 0; i < argc; i++)
        printf("%s%s", i ? "|" : "", argv[i] ? argv[i] : "NULL");
    printf("\n");
    return 0;
}

static void query(const char *label, const char *sql) {
    printf("-- %s\n", label);
    int rc = sqlite3_exec(db, sql, print_row, NULL, &err);
    if (rc != SQLITE_OK) {
        printf("FAIL: %s\n", err ? err : "?");
        sqlite3_free(err);
        err = NULL;
    }
}

int main(void) {
    if (sqlite3_open(":memory:", &db) != SQLITE_OK) {
        printf("FAIL: open\n");
        return 1;
    }

    exec_ok("CREATE TABLE authors(id INTEGER PRIMARY KEY, name TEXT, country TEXT);");
    exec_ok("CREATE TABLE books(id INTEGER PRIMARY KEY, author_id INTEGER, title TEXT, year INTEGER, price REAL);");
    exec_ok("CREATE TABLE sales(book_id INTEGER, qty INTEGER, day TEXT);");

    // Deterministic pseudo-random data (no rand() — same sequence everywhere).
    exec_ok("BEGIN;");
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db, "INSERT INTO authors VALUES(?1,?2,?3);", -1, &st, NULL);
    const char *countries[] = {"US", "GB", "FR", "DE", "JP"};
    for (int i = 1; i <= 50; i++) {
        char name[32];
        snprintf(name, sizeof(name), "author_%02d", i);
        sqlite3_bind_int(st, 1, i);
        sqlite3_bind_text(st, 2, name, -1, SQLITE_STATIC);
        sqlite3_bind_text(st, 3, countries[(i * 7 + 3) % 5], -1, SQLITE_STATIC);
        sqlite3_step(st);
        sqlite3_reset(st);
    }
    sqlite3_finalize(st);

    sqlite3_prepare_v2(db, "INSERT INTO books VALUES(?1,?2,?3,?4,?5);", -1, &st, NULL);
    for (int i = 1; i <= 400; i++) {
        char title[32];
        snprintf(title, sizeof(title), "book_%03d", i);
        sqlite3_bind_int(st, 1, i);
        sqlite3_bind_int(st, 2, (i * 13 + 7) % 50 + 1);
        sqlite3_bind_text(st, 3, title, -1, SQLITE_STATIC);
        sqlite3_bind_int(st, 4, 1900 + (i * 37) % 125);
        sqlite3_bind_double(st, 5, 5.0 + (i * 11) % 90 + 0.25 * (i % 4));
        sqlite3_step(st);
        sqlite3_reset(st);
    }
    sqlite3_finalize(st);

    sqlite3_prepare_v2(db, "INSERT INTO sales VALUES(?1,?2,?3);", -1, &st, NULL);
    const char *days[] = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
    for (int i = 1; i <= 2000; i++) {
        sqlite3_bind_int(st, 1, (i * 29 + 11) % 400 + 1);
        sqlite3_bind_int(st, 2, (i * 7 + 1) % 20 + 1);
        sqlite3_bind_text(st, 3, days[(i * 3) % 7], -1, SQLITE_STATIC);
        sqlite3_step(st);
        sqlite3_reset(st);
    }
    sqlite3_finalize(st);
    exec_ok("COMMIT;");

    exec_ok("CREATE INDEX idx_books_author ON books(author_id);");
    exec_ok("CREATE INDEX idx_sales_book ON sales(book_id);");
    exec_ok("CREATE INDEX idx_books_year ON books(year);");

    query("counts", "SELECT (SELECT count(*) FROM authors),"
                    "       (SELECT count(*) FROM books),"
                    "       (SELECT count(*) FROM sales);");

    query("join+group",
          "SELECT a.country, count(*), sum(b.price) "
          "FROM authors a JOIN books b ON b.author_id = a.id "
          "GROUP BY a.country ORDER BY a.country;");

    query("having",
          "SELECT author_id, count(*) c FROM books "
          "GROUP BY author_id HAVING c > 8 ORDER BY author_id LIMIT 5;");

    query("correlated subquery",
          "SELECT name, (SELECT count(*) FROM books b WHERE b.author_id = a.id) "
          "FROM authors a WHERE a.id <= 5 ORDER BY a.id;");

    query("join three tables",
          "SELECT a.name, sum(s.qty) "
          "FROM sales s JOIN books b ON b.id = s.book_id "
          "             JOIN authors a ON a.id = b.author_id "
          "WHERE a.country = 'US' GROUP BY a.name ORDER BY a.name LIMIT 5;");

    query("aggregate stats",
          "SELECT min(year), max(year), avg(price), total(qty) "
          "FROM books CROSS JOIN sales ON sales.book_id = books.id;");

    // Transactions with rollback.
    exec_ok("BEGIN;");
    exec_ok("UPDATE books SET price = price * 2 WHERE year < 1925;");
    query("inside txn", "SELECT count(*), sum(price) FROM books WHERE year < 1925;");
    exec_ok("ROLLBACK;");
    query("after rollback", "SELECT count(*), sum(price) FROM books WHERE year < 1925;");

    exec_ok("BEGIN;");
    exec_ok("DELETE FROM sales WHERE day = 'Sun';");
    exec_ok("UPDATE sales SET qty = qty + 1 WHERE day = 'Sat';");
    exec_ok("COMMIT;");
    query("after delete/update", "SELECT day, sum(qty) FROM sales GROUP BY day ORDER BY day;");

    query("text ops",
          "SELECT upper(substr(title, 1, 4)), length(title) FROM books WHERE id <= 3;");

    query("integer/real math",
          "SELECT sum(price * qty), cast(avg(qty) * 100 as int) "
          "FROM sales JOIN books ON books.id = sales.book_id;");

    sqlite3_close(db);
    printf("DONE\n");
    return 0;
}
