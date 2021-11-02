//
// Created by Ben_Sepanski on 10/15/2021.
//

#ifndef BRICK_TEMPLATE_UTILS_H
#define BRICK_TEMPLATE_UTILS_H

#include <algorithm>
#include <array>

// Some common utilities based on CUDA
#ifdef __CUDACC__
#define IF_CUDA_ELSE(CudaCase, NoCudaCase) CudaCase
#define ARRAY_TYPE(Type, Size) Type[Size]
#else
#define IF_CUDA_ELSE(CudaCase, NoCudaCase) NoCudaCase
#define ARRAY_TYPE(Type, Size) std::array<Type, Size>
#endif

namespace brick {
  namespace templateutils {

    /**
     * Used to manipulate values from a parameter pack
     * @tparam T the type of elements in the parameter pack
     */
    template<typename T>
    struct ParameterPackManipulator {
      /**
       * Empty struct to hold a parameter pack
       * @tparam Elements elements of the parameter pack
       */
      template<T ... Elements>
      struct Pack;

      /**
       * Generic template for specialization
       */
      template<typename Pack>
      struct PackToArray;

      /**
       * @tparam Elements the elements of the pack
       */
      template<T ... Elements>
      struct PackToArray<Pack<Elements...> > {
        /**
         * Elements of the pack as an array
         */
        static constexpr std::array<T, sizeof...(Elements)> value = { Elements... };
      };

      /**
       * Generic template for specialization
       *
       * Appends packs
       */
      template<typename ... Packs>
      struct PackAppender;

      /**
       * >= 3 packs case
       * @tparam HeadPack the first pack
       * @tparam TailPacks the remaining packs
       */
      template<typename HeadPack, typename ... TailPacks>
      struct PackAppender<HeadPack, TailPacks...> {
        /** The pack holding the appended values */
        typedef typename PackAppender<HeadPack, typename PackAppender<TailPacks...>::type>::type type;
      };

      /**
       * two-packs case
       * @tparam Pack1Values values in the first pack
       * @tparam Pack2Values values in the second pack
       */
      template<T ... Pack1Values, T ... Pack2Values>
      struct PackAppender<Pack<Pack1Values...>, Pack<Pack2Values...> >{
        /** The pack holding the appended values */
        typedef Pack<Pack1Values..., Pack2Values...> type;
      };

      /**
       * one-pack case
       * @tparam PackValues the values in the pack
       */
      template<T ... PackValues>
      struct PackAppender<Pack<PackValues...> > {
        /** The pack holding the appended values */
        typedef Pack<PackValues...> type;
      };

      /**
       * Reverses values in the pack. Empty pack case
       * (non-empty handled via partial template specialization)
       * @tparam EmptyPack the empty pack
       */
      template<typename EmptyPack>
      struct PackReverser {
        /** The reversed pack */
        typedef Pack<> type;
      };

      /**
       * Non-empty pack case
       * @tparam FirstValue the first value in the pack
       * @tparam TailValues the remaining values in the pack
       */
      template<T FirstValue, T ... TailValues>
      struct PackReverser<Pack<FirstValue, TailValues...> > {
        typedef typename PackReverser<Pack<TailValues...> >::type reversedTail;
        /** The reversed pack */
        typedef typename PackAppender<reversedTail, Pack<FirstValue> >::type type;
      };

      /**
       * General template for specialization
       * Repeat Pack NumRepeats number of time
       */
      template<typename Pack, unsigned NumRepeats>
      struct PackRepeater;

      /**
       * NumRepeats > 0 case
       * @tparam NumRepeats number of times to repeat
       * @tparam PackValues values of the pack
       */
      template<unsigned NumRepeats, T ... PackValues>
      struct PackRepeater<Pack<PackValues...>, NumRepeats> {
        /** Pack repeated NumRepeats times */
        typedef typename PackAppender<
            typename PackRepeater<Pack<PackValues...>, NumRepeats - 1>::type,
            Pack<PackValues...>
                >::type type;
      };

      /**
       * NumRepeats == 0 case
       * @tparam PackValues the values in the pack
       */
      template<T ... PackValues>
      struct PackRepeater<Pack<PackValues...>, 0> {
        /** Empty pack representing repeating the pack 0 times */
        typedef Pack<> type;
      };

      /**
       * Generic template for specialization
       * @tparam N the length to pad up to
       * @tparam PaddingValue the value to pad with
       * @tparam PackToPad the pack to pad
       */
      template<size_t N, T PaddingValue, typename PackToPad>
      struct PackPadder;

      /**
       * @tparam PackValues the values in the pack
       */
      template<size_t N, T PaddingValue, T ... PackValues>
      struct PackPadder<N, PaddingValue, Pack<PackValues...> > {
        static constexpr unsigned NumPaddedElements = sizeof...(PackValues) <= N
                                                    ? N - sizeof...(PackValues)
                                                    : 0;
        static constexpr unsigned NumTotalElements = sizeof...(PackValues) + NumPaddedElements;
        typedef typename PackAppender<
            Pack<PackValues...>,
            typename PackRepeater<Pack<PaddingValue>, NumPaddedElements>::type
            >::type type;
      };

      /// Functions to encapsulate common operations so that users can
      /// avoid dealing with ugly types

      /**
       * @return An array of the pack values in reverse
       */
      template<T ... PackValues>
      static constexpr std::array<T, sizeof...(PackValues)> reverse() {
        return PackToArray<
            typename PackReverser<Pack<PackValues...>>::type
            >::value;
      }

      /**
       * @return an array of the pack values in reverse, after being
       *         right-padded with PaddingValue up to length N
       */
      template<size_t N, T PaddingValue, T ... PackValues>
      static constexpr std::array<T, PackPadder<N, PaddingValue, Pack<PackValues...> >::NumTotalElements >
      padAndReverse() {
        return PackToArray<
            typename PackReverser<
                typename PackPadder<N,
                                    PaddingValue,
                                    Pack<PackValues...>>::type
                >::type
            >::value;
      }
    };

    /// Tools to deal with Unsigned types

    /**
     * Has a public typedef of type Pack<0,...,ExclusiveEnd-1>
     *
     * ExclusiveEnd > 0 case is implemented here
     *
     * @tparam ExclusiveEnd The exclusive end of the sequence
     */
    template<unsigned ExclusiveEnd>
    struct UnsignedIndexSequence{
    private:
      template<typename = typename UnsignedIndexSequence<ExclusiveEnd - 1>::type >
      struct SequenceGenerator;

      template<unsigned ... Elements>
      struct SequenceGenerator<typename ParameterPackManipulator<unsigned>::Pack<Elements...> > {
        typedef typename ParameterPackManipulator<unsigned>::Pack<Elements..., ExclusiveEnd - 1> type;
      };

    public:
      /** The pack Pack<0,...,ExclusiveEnd-1> */
      typedef typename SequenceGenerator<>::type type;
    };

    /**
     * 0-case
     */
    template<>
    struct UnsignedIndexSequence<0> {
      /** The pack Pack<0,...,N-1> is the empty pack for N = 0*/
      typedef typename ParameterPackManipulator<unsigned>::Pack<> type;
    };

    /// Reduction

    /**
       * Reduce a list (empty list case)
       * @see reduce
     */
    template<typename AggregatorFunction, typename OutputType>
    constexpr OutputType reduce(AggregatorFunction f, OutputType identity) {
      return identity;
    }

    /**
       * Reduce a list of type T (singleton list case)
       * @see reduce
     */
    template<typename AggregatorFunction, typename OutputType, typename T>
    constexpr OutputType reduce(AggregatorFunction f, OutputType identity, T singleton) {
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
    constexpr OutputType reduce(AggregatorFunction f, OutputType identity, T1 first, T1 second, T ... tail) {
      return reduce(f, identity, f(first, second), tail...);
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
