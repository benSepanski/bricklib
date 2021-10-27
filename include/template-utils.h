//
// Created by Ben_Sepanski on 10/15/2021.
//

#ifndef BRICK_TEMPLATE_UTILS_H
#define BRICK_TEMPLATE_UTILS_H

#include <algorithm>
#include <array>

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
        typedef typename PackAppender<
            Pack<PackValues...>,
            typename PackRepeater<Pack<PaddingValue>, NumPaddedElements>::type
            >::type type;
      };
    };

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
