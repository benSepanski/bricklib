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
      struct Pack{
        static constexpr size_t size = sizeof...(Elements);
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
       * Used to pop the head off of a pack
       * @tparam Pack the pack
       * @see PackPopper
       */
      template<typename Pack>
      struct PackPopper;

      /**
       * @tparam Head head of the pack
       * @tparam Tail tail of the pack
       * @see PackPopper
       */
      template<T Head, T ... Tail>
      struct PackPopper<Pack<Head, Tail...> > {
        typedef Pack<Tail...> type;
        static constexpr T value = Head;
      };

      /// Functions to encapsulate common operations so that users can
      /// avoid dealing with ugly types

      /**
       * @note this implementation is the Pack::size >= 2 case
       * @tparam Pack the parameter pack
       * @param index the index
       * @return the index th value in Pack
       * @see get
       */
      template<typename Pack>
      static constexpr
      typename std::enable_if<1 < Pack::size, T>::type get(size_t index) {
        typedef PackPopper<Pack> PoppedPack;
        return index == 0
               ? PoppedPack::value
               : get<typename PoppedPack::type>(index - 1);
      }

      /**
       * @note this implementation is the Pack::size < 2 case
       * @see get
       */
      template<typename Pack>
      static constexpr
      typename std::enable_if<Pack::size <= 1, T>::type get(size_t) {
        static_assert(Pack::size > 0, "Index out of bounds");
        typedef PackPopper<Pack> PoppedPack;
        return PoppedPack::value;
      }

      /**
       * @note The empty pack case is implemented separately to avoid a compiler
       *       warning
       * @tparam P the parameter pack
       * @tparam DefaultValue the default value
       * @param index the index to get
       * @return the defaultValue if index is out-of-bounds, otherwise the
       *         index th value in P
       * @see getOrDefault
       */
      template<typename P, T DefaultValue>
      static constexpr
      typename std::enable_if<P::size != 0, T>::type getOrDefault(size_t index) {
        // Pad pack with default value to avoid out-of-bounds access in
        // the case where P is the empty pack.
        typedef typename PackAppender<P, Pack<DefaultValue> >::type PaddedPack;
        return (index < P::size)
            ? get<PaddedPack>(index >= P::size ? 0 : index)
            : DefaultValue;
      }

      /**
       * @note implementation for empty pack case
       * @see getOrDefault
       */
      template<typename P, T DefaultValue>
      static constexpr
      typename std::enable_if<P::size == 0, T>::type getOrDefault(size_t) {
        return DefaultValue;
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
      constexpr inline R callOnReversedArgs(F &&f, Tuple && args, ParameterPackManipulator<unsigned>::Pack<Range0ToNumArgs...>) {
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
                                   typename UnsignedIndexSequence<(unsigned) sizeof...(ArgTypes)>::type());
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
