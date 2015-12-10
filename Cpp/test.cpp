# include <iostream>

template <int N, int D> struct Frak {
        static const long Num = N;
        static const long Den = D;
};
  
template <int N, typename F> struct ScalarMultiplication {
    typedef Frak<N*F::Num, F::Den> result;
};
  
template <int X, int Y>   struct MCD {
        static const long result = MCD<Y, X % Y>::result;
};
  
template <int X> struct MCD<X, 0> {
        static const long result = X;
};
  
template <class F> struct Simpl {
        static const long mcd = MCD<F::Num, F::Den>::result;
        typedef Frak< F::Num / mcd, F::Den / mcd > result;
};
  
template <typename X1, typename Y1> struct SameBase {
    typedef typename ScalarMultiplication< Y1::Den, X1>::result X;
    typedef typename ScalarMultiplication< X1::Den, Y1>::result Y;
};
  
template <typename X, typename Y> struct Sum {
    typedef SameBase<X, Y> B;
    static const long Num = B::X::Num + B::Y::Num;
    static const long Den = B::Y::Den; // == B::X::Den
    typedef typename Simpl< Frak<Num, Den> >::result result;
};
  
template <int N> struct Fact {
    static const long result = N * Fact<N-1>::result;
};
template <> struct Fact<0> {
    static const long result = 1;
};
  
template <int N> struct E {
    // e = S(1/n!) = 1/0! + 1/1! + 1/2! + ...
    static const long Den = Fact<N>::result;
    typedef Frak< 1, Den > term;
    typedef typename E<N-1>::result next_term;
    typedef typename Sum< term, next_term >::result result;
};

template <> struct E<0> {
    typedef Frak<1, 1> result;
};
  
int main() {
    typedef E<8>::result X;
    std::cout << "e = " << (1.0 * X::Num / X::Den) << "\n";
    std::cout << "e = " << X::Num <<"/"<< X::Den << "\n";
    return 0;
}