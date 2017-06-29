#ifndef PGBART_COMPARE_HPP
#define PGBART_COMPARE_HPP

#include <iostream>
#include <string>

/**************************************
File name : compare.hpp
Date : 2016-12-7
Function List : compare_if_zero(T value)
				compare_if_equal(T a, T b)
				compare_if(T a, string compare_symbol, T b) ; symbol can be ==, !=, >, >=, <, <=
				compare_if_between(T min, T medium, T max)
***************************************/

using std::string;

namespace pgbart {
	namespace compare {

		template<typename T>
		bool compare_if_zero(T value) {
			return value < 1e-10 && value > -1e-10;
		}

		template<typename T>
		bool compare_if_equal(T a, T b) {
			return compare_if_zero(a - b);
		}

		template<typename T>
		bool compare_if(T a, string compare_symbol, T b) {
			if (compare_symbol == "==") {
				return a == b;
			}
			else if (compare_symbol == "!=") {
				return a != b;
			}
			else if (compare_symbol == ">") {
				return a > b;
			}
			else if (compare_symbol == ">=") {
				return a >= b;
			}
			else if (compare_symbol == "<") {
				return a < b;
			}
			else if (compare_symbol == "<=") {
				return a <= b;
			}
			else {
				std::cout << "the 2nd parameter must be include in {\"==\", \"!=\", \">\", \">=\", \"<\",\"<=\"}\n" << std::endl;
				exit(1);
			}
		}

		template<typename T>
		bool compare_if_between(T min, T medium, T max) {
			return compare_if(min, "<=", medium) && compare_if(medium, "<=", max);
		}
	}
}

#endif
