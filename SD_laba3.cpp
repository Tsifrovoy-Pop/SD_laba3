#include <iostream>
#include <vector>
#include <stack>
using namespace std;

int main() {
    setlocale(LC_ALL, "Russian");
    system("chcp 65001 > nul");

    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << "Введите количетво элементов: \n";
    int N;
    if (!(cin >> N)) return 0;
    vector<long long> A(N);
    cout << "Введите элементы: \n";
    for (int i = 0; i < N; ++i) cin >> A[i];

    vector<long long> ans(N);
    stack<long long> st;

    for (int i = N - 1; i >= 0; --i) {
        while (!st.empty() && st.top() <= A[i]) {
            st.pop();
        }
        if (st.empty()) ans[i] = 0;
        else ans[i] = st.top();
        st.push(A[i]);
    }
    cout << "Ответ: \n";
    for (int i = 0; i < N; ++i) {
        cout << ans[i];
        if (i + 1 < N) cout << ' ';
    }

    cout << "\n020303-АИСа-025 Динер З.М.";

    return 0;
}