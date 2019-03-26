#include <iostream>
#include <vector>
#include <string>

using namespace std;

int min(int a, int b, int c) {
    return a < b ? a : (b < c ? b : c);
}

void print_table2d(vector<vector<int>> table) {
    int rows = table.size();
    int cols = table[0].size();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf ("%d ", table[i][j]);
        }
        printf ("\n");
    }
}

int edit_distance(string s1, string s2, int csub=1, int cdel=1, int cins=1, bool ptable=false) {
    int n = s1.length();
    int m = s2.length();
    
    vector<vector<int>> table (n+1, vector<int> (m+1, 0));

    for (int i = 1; i <= n; ++i)
        table[i][0] = table[i-1][0] + 1;

    for (int i = 1; i <= m; ++i)
        table[0][i] = table[0][i-1] + 1;
    
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (s1[i-1] == s2[j-1]) {
                table[i][j] = table[i-1][j-1];
            } else {
                table[i][j] = min(table[i-1][j] + cins, \
                              table[i][j-1] + cdel, \
                              table[i-1][j-1] + csub);
            }
        }
    }
    
    // print table
    if (ptable)
        print_table2d(table);

    return table[n][m];
}

int main() {
    
    string s1 = "inter";
    string s2 = "exter";

    cout << edit_distance(s1, s2, 1, 1, 1, 1) << "\n";

    return 0;
}
