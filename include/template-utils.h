//
// Created by Ben_Sepanski on 10/15/2021.
//

#ifndef BRICK_TEMPLATE_UTILS_H
#define BRICK_TEMPLATE_UTILS_H

#include <algorithm>
#include <array>
#include <tuple>

// Some common utilities based on CUDA
#ifdef __CUDACC__
#define IF_CUDA_ELSE(CudaCase, NoCudaCase) CudaCase
#else
#define IF_CUDA_ELSE(CudaCase, NoCudaCase) NoCudaCase
#endif

namespace brick {
  namespace templateutils {

    /// Some function utils

    /**
       * Reduce a list (empty list case)
       * @see reduce
     */
    template<typename AggregatorFunction, typename OutputType>
    constexpr inline OutputType reduce(AggregatorFunction, OutputType identity) {
      return identity;
    }

    /**
       * Reduce a list of type T (singleton list case)
       * @see reduce
     */
    template<typename AggregatorFunction, typename OutputType, typename T>
    constexpr inline OutputType reduce(AggregatorFunction f, OutputType identity, T singleton) {
      return f(identity, singleton);
    }

    /**
       * Reduce a list of type T
       *
       * @tparam AggregatorFunction type of the aggregator : OutputType x T -> OutputType
       * @tparam OutputType type of the output
       * @tparam T1 type of first two types
       * @tparam T parameter pack for tail
       * @param f the reducer function
       * @param identity identity
       * @param first first item to be reduced
       * @param second second item to be reduced
       * @param tail... the remainder to be reduced
       * @return the reduced value
     */
    template<typename AggregatorFunction, typename OutputType, typename T1, typename ... T>
    constexpr inline OutputType reduce(AggregatorFunction f, OutputType identity, T1 first, T1 second, T ... tail) {
      return reduce(f, identity, f(first, second), tail...);
    }

    namespace { // Begin anonymous namespace
      /**
       * Implementation of calling f with reversed arguments based on
       * https://gist.github.com/SephDB/a084c2a8cce378b3cdea502c233d2f4a
       */
      template<typename R, typename F, typename Tuple, unsigned ... Range0ToNumArgs>
      constexpr inline R callOnReversedArgs(F &&f, Tuple && args, std::integer_sequence<unsigned, Range0ToNumArgs...>) {
        constexpr size_t N = std::tuple_size<Tuple>::value;
        return f(std::get<N - 1 - Range0ToNumArgs>(args)...);
      }
    } // end anonymous namespace

    /**
     *
     * @tparam R the return type of f
     * @tparam F the function type of f
     * @tparam ArgTypes the types of the function arguments
     * @param f the function to apply
     * @param args the arguments to reverse, and then apply f to
     * @return f(argN-1,argN-2,...,arg1,arg0)
     */
    template<typename R, typename F, typename ... ArgTypes>
    constexpr inline R callOnReversedArgs(F &&f, ArgTypes&& ... args) {
      return callOnReversedArgs<R>(f, std::forward_as_tuple(args...),
                                   std::make_integer_sequence<unsigned, sizeof...(ArgTypes)>());
    }

    /// Tools to deal with boolean types

    /**
     * Check if all Bools are true at compile-time
     * @tparam Bools the booleans
     */
    template<bool ... Bools>
    struct All {
      /** Case where sizeof...(Bools) == 0, value is true*/
      static constexpr bool value = true;
    };

    /**
     * @tparam HeadBool the first boolean value
     * @tparam TailBools the remaining boolean values
     * @see All
     */
    template<bool HeadBool, bool ... TailBools>
    struct All<HeadBool, TailBools...> {
      /** Case where sizeof...(Bools) > 0, value is reduction of bools using And */
      static constexpr bool value = HeadBool && All<TailBools...>::value;
    };
  }
}

#endif // BRICK_TEMPLATE_UTILS_H
