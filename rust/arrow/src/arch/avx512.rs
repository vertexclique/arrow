// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::array::*;
use crate::datatypes::{ArrowPrimitiveType, ArrowNativeType};

pub(crate) const AVX512_U8X64_LANES: usize = 64;

///
/// AVX-512 bit and implementation
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn avx512_bin_and(left: &[u8], right: &[u8], res: &mut [u8]) {
    use core::arch::x86_64::{__m512i, _mm512_and_si512, _mm512_loadu_epi64};

    let l: __m512i = _mm512_loadu_epi64(left.as_ptr() as *const _);
    let r: __m512i = _mm512_loadu_epi64(right.as_ptr() as *const _);
    let f = _mm512_and_si512(l, r);
    let s = &f as *const __m512i as *const u8;
    let d = res.get_unchecked_mut(0) as *mut _ as *mut u8;
    std::ptr::copy_nonoverlapping(s, d, std::mem::size_of::<__m512i>());
}

///
/// AVX-512 bit or implementation
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn avx512_bin_or(left: &[u8], right: &[u8], res: &mut [u8]) {
    use core::arch::x86_64::{__m512i, _mm512_loadu_epi64, _mm512_or_si512};

    let l: __m512i = _mm512_loadu_epi64(left.as_ptr() as *const _);
    let r: __m512i = _mm512_loadu_epi64(right.as_ptr() as *const _);
    let f = _mm512_or_si512(l, r);
    let s = &f as *const __m512i as *const u8;
    let d = res.get_unchecked_mut(0) as *mut _ as *mut u8;
    std::ptr::copy_nonoverlapping(s, d, std::mem::size_of::<__m512i>());
}

///
/// Sorting network for a single SIMD vector of i64s
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn avx512_vec_sort_i64_single<'a>(input: &[i64]) -> [i64; 8] {
    use core::arch::x86_64::{
        __m512i, _mm512_loadu_epi64, _mm512_mask_mov_epi64, _mm512_max_epi64,
        _mm512_min_epi64, _mm512_permutexvar_epi64, _mm512_set_epi64,
    };

    // First wiring's permute exchange for the sorting network
    let mut inp: __m512i = _mm512_loadu_epi64(input.as_ptr() as *const _);
    let idxnn1: __m512i = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
    let wire_n: __m512i = _mm512_permutexvar_epi64(idxnn1, inp);
    let wire_n_min: __m512i = _mm512_min_epi64(wire_n, inp);
    let wire_n_max: __m512i = _mm512_max_epi64(wire_n, inp);
    inp = _mm512_mask_mov_epi64(wire_n_min, 0xAA, wire_n_max);

    // Second wiring's permute exchange for the sorting network
    let idxnn2: __m512i = _mm512_set_epi64(4, 5, 6, 7, 0, 1, 2, 3);
    let wire_n: __m512i = _mm512_permutexvar_epi64(idxnn2, inp);
    let wire_n_min: __m512i = _mm512_min_epi64(wire_n, inp);
    let wire_n_max: __m512i = _mm512_max_epi64(wire_n, inp);
    inp = _mm512_mask_mov_epi64(wire_n_min, 0xCC, wire_n_max);

    // Third wiring's permute exchange for the sorting network
    let idxnn3: __m512i = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
    let wire_n: __m512i = _mm512_permutexvar_epi64(idxnn3, inp);
    let wire_n_min: __m512i = _mm512_min_epi64(wire_n, inp);
    let wire_n_max: __m512i = _mm512_max_epi64(wire_n, inp);
    inp = _mm512_mask_mov_epi64(wire_n_min, 0xAA, wire_n_max);

    // Fourth wiring's permute exchange, does forwarding.
    let idxnn4: __m512i = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    let wire_n: __m512i = _mm512_permutexvar_epi64(idxnn4, inp);
    let wire_n_min: __m512i = _mm512_min_epi64(wire_n, inp);
    let wire_n_max: __m512i = _mm512_max_epi64(wire_n, inp);
    inp = _mm512_mask_mov_epi64(wire_n_min, 0xF0, wire_n_max);

    // Fifth wiring's permute exchange for the sorting network
    let idxnn5: __m512i = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
    let wire_n: __m512i = _mm512_permutexvar_epi64(idxnn5, inp);
    let wire_n_min: __m512i = _mm512_min_epi64(wire_n, inp);
    let wire_n_max: __m512i = _mm512_max_epi64(wire_n, inp);
    inp = _mm512_mask_mov_epi64(wire_n_min, 0xCC, wire_n_max);

    // Sixth wiring's permute exchange for the sorting network
    let idxnn6: __m512i = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
    let wire_n: __m512i = _mm512_permutexvar_epi64(idxnn6, inp);
    let wire_n_min: __m512i = _mm512_min_epi64(wire_n, inp);
    let wire_n_max: __m512i = _mm512_max_epi64(wire_n, inp);
    inp = _mm512_mask_mov_epi64(wire_n_min, 0xAA, wire_n_max);

    std::mem::transmute(inp)
}

///
/// Sorting network with SIMD merger for two SIMD vector of i64s
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn avx512_vec_sort_i64_double(
    left: &[i64],
    right: &[i64],
) -> [[i64; 8]; 2] {
    use core::arch::x86_64::{
        __m512i, _mm512_loadu_epi64, _mm512_mask_mov_epi64, _mm512_max_epi64,
        _mm512_min_epi64, _mm512_permutexvar_epi64, _mm512_set_epi64,
    };

    let (l, r) = (
        avx512_vec_sort_i64_single(left),
        avx512_vec_sort_i64_single(right),
    );

    let mut l: __m512i = _mm512_loadu_epi64(l.as_ptr() as *const _);
    let mut r: __m512i = _mm512_loadu_epi64(r.as_ptr() as *const _);

    // Full blend of the both vector wires
    let idxnn1: __m512i = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    let wire_n: __m512i = _mm512_permutexvar_epi64(idxnn1, l);
    l = _mm512_min_epi64(r, wire_n);
    r = _mm512_max_epi64(r, wire_n);

    // Carries on with normal sorting network operation
    let idxnn2: __m512i = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    let wire_n: __m512i = _mm512_permutexvar_epi64(idxnn2, l);
    let wire_n_min: __m512i = _mm512_min_epi64(wire_n, l);
    let wire_n_max: __m512i = _mm512_max_epi64(wire_n, l);
    l = _mm512_mask_mov_epi64(wire_n_min, 0xF0, wire_n_max); // 0x33
    let wire_n: __m512i = _mm512_permutexvar_epi64(idxnn2, r);
    let wire_n_min: __m512i = _mm512_min_epi64(wire_n, r);
    let wire_n_max: __m512i = _mm512_max_epi64(wire_n, r);
    r = _mm512_mask_mov_epi64(wire_n_min, 0xF0, wire_n_max); // 0x33

    let idxnn3: __m512i = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
    let wire_n: __m512i = _mm512_permutexvar_epi64(idxnn3, l);
    let wire_n_min: __m512i = _mm512_min_epi64(wire_n, l);
    let wire_n_max: __m512i = _mm512_max_epi64(wire_n, l);
    l = _mm512_mask_mov_epi64(wire_n_min, 0xCC, wire_n_max);
    let wire_n: __m512i = _mm512_permutexvar_epi64(idxnn3, r);
    let wire_n_min: __m512i = _mm512_min_epi64(wire_n, r);
    let wire_n_max: __m512i = _mm512_max_epi64(wire_n, r);
    r = _mm512_mask_mov_epi64(wire_n_min, 0xCC, wire_n_max);

    let idxnn4: __m512i = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
    let wire_n: __m512i = _mm512_permutexvar_epi64(idxnn4, l);
    let wire_n_min: __m512i = _mm512_min_epi64(wire_n, l);
    let wire_n_max: __m512i = _mm512_max_epi64(wire_n, l);
    l = _mm512_mask_mov_epi64(wire_n_min, 0xAA, wire_n_max);
    let wire_n: __m512i = _mm512_permutexvar_epi64(idxnn4, r);
    let wire_n_min: __m512i = _mm512_min_epi64(wire_n, r);
    let wire_n_max: __m512i = _mm512_max_epi64(wire_n, r);
    r = _mm512_mask_mov_epi64(wire_n_min, 0xAA, wire_n_max);

    let lf: [i64; 8] = std::mem::transmute(l);
    let rf: [i64; 8] = std::mem::transmute(r);

    [lf, rf]
}

///
/// Permute exchange width for the AVX-512 SIMD application
pub(crate) const PERMUTE_EXCHANGE_WIDTH: usize = 8;

///
/// Merge layer for sorting network
fn merger_net(mut input: &mut [i64], mark: usize, slice: usize) -> &mut [i64] {
    let half = slice / 2;
    if slice > 1 {
        (mark..half+mark).into_iter().for_each(|e| unsafe {
            if input[e] > input[e + half] {
                let pl: *mut i64 = &mut input[e];
                let pr: *mut i64 = &mut input[e + half];
                std::ptr::swap(pl, pr);
            }
        });
        merger_net(&mut input, mark, half);
        merger_net(&mut input, half+mark, half);
    }
    input
}

///
/// Cold path marker for hinting the CPU for the further optimizations.
#[inline]
#[cold]
fn cold() {}

///
/// Size independent sorter for any vector which is power of two.
pub(crate) unsafe fn avx512_vec_sort_i64(input: &[i64]) -> Vec<i64> {
    if (input.len() / 2) == PERMUTE_EXCHANGE_WIDTH {
        let v: Vec<&[i64]> = input.chunks_exact(PERMUTE_EXCHANGE_WIDTH).collect();
        let x = avx512_vec_sort_i64_double(&v[0], &v[1]);
        [x[0], x[1]].concat()
    } else {
        if (input.len() / 2) == 0 {
            cold();
            input.to_vec()
        } else {
            let mut it = input.chunks_exact(input.len() / 2);
            let l = avx512_vec_sort_i64(it.next().unwrap());
            let mut r = avx512_vec_sort_i64(it.next().unwrap());
            r.reverse();
            let mut c = [l, r].concat();
            let c = c.as_mut_slice();
            merger_net(c, 0, c.len()).to_vec()
        }
    }
}

pub(crate) fn avx512_kv_sort<T>(values: &PrimitiveArray<T>, value_indices: Vec<u32>) -> Vec<(u32, T::Native)>
where
    T: ArrowPrimitiveType
{
    let value_data = value_indices
        .iter()
        .map(|index| {
            *index as i64 | (values.value(*index as usize).cast::<i64>().unwrap() << 32)
        })
        .collect::<Vec<i64>>();

    unsafe { avx512_vec_sort_i64(value_data.as_slice()) }.iter().map(|e| {
        let i = ((*e << 32) >> 32) as u32;
        let v = T::Native::from_isize(((*e ^ (i as i64)) >> 32) as isize).unwrap();
        (i, v)
    })
    .collect::<Vec<(u32, T::Native)>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitwise_and_avx512() {
        let buf1 = [0b00110011u8; 64];
        let buf2 = [0b11110000u8; 64];
        let mut buf3 = [0b00000000; 64];
        unsafe {
            avx512_bin_and(&buf1, &buf2, &mut buf3);
        };
        for i in buf3.iter() {
            assert_eq!(&0b00110000u8, i);
        }
    }

    #[test]
    fn test_bitwise_or_avx512() {
        let buf1 = [0b00010011u8; 64];
        let buf2 = [0b11100000u8; 64];
        let mut buf3 = [0b00000000; 64];
        unsafe {
            avx512_bin_or(&buf1, &buf2, &mut buf3);
        };
        for i in buf3.iter() {
            assert_eq!(&0b11110011u8, i);
        }
    }

    #[test]
    fn test_vec_sort_i64_single_avx512() {
        let buf1 = [9_i64, 4, 2, 0, 1, 10, 8, 7];
        let res = unsafe { avx512_vec_sort_i64_single(&buf1) };
        assert_eq!(res, [0_i64, 1, 2, 4, 7, 8, 9, 10]);
    }

    #[test]
    fn test_vec_sort_i64_double_avx512() {
        let buf1 = [9_i64, 4, 2, 0, 1, 10, 8, 7];
        let buf2 = [3_i64, 71, 13, 100, 67, 5, -1, 89];
        let res = unsafe { avx512_vec_sort_i64_double(&buf1, &buf2) };
        assert_eq!(
            res,
            [[-1, 0, 1, 2, 3, 4, 5, 7], [8, 9, 10, 13, 67, 71, 89, 100]]
        );
    }

    #[test]
    fn test_vec_sort_i64_avx512_bounds() {
        let buf1 = (i64::min_value()..i64::min_value()+16).into_iter().rev().collect::<Vec<i64>>();
        let res = unsafe { avx512_vec_sort_i64(&buf1) };
        assert_eq!(res, (i64::min_value()..i64::min_value()+16).into_iter().collect::<Vec<i64>>());
    }

    #[test]
    fn test_vec_sort_i64_avx512() {
        let buf1 = (0..128_i64).into_iter().rev().collect::<Vec<i64>>();
        let res = unsafe { avx512_vec_sort_i64(&buf1) };
        assert_eq!(res, (0..128_i64).into_iter().collect::<Vec<i64>>());

        let buf1 = (0..8192_i64).into_iter().rev().collect::<Vec<i64>>();
        let res = unsafe { avx512_vec_sort_i64(&buf1) };
        assert_eq!(res, (0..8192_i64).into_iter().collect::<Vec<i64>>());
    }

    #[test]
    fn test_vec_sort_i64_avx512_redundant() {
        let mut buf1 = [1, 2, 3, 4_i64]
            .repeat(2_i32.pow(18) as usize)
            .iter()
            .map(|e| *e)
            .collect::<Vec<i64>>();

        let res = unsafe { avx512_vec_sort_i64(&buf1) };

        buf1.sort();
        assert_eq!(res, buf1);
    }

    #[test]
    fn test_vec_sort_i8_array_avx512() {
        let mut kvs = [1, 2, 3, 4_i8]
            .repeat(2_i32.pow(18) as usize)
            .iter()
            .enumerate()
            .map(|(idx, e)| (idx as u32, *e))
            .collect::<Vec<(u32, i8)>>();

        let idxs = kvs.iter().map(|(k, _)| *k).collect::<Vec<u32>>();
        let values = kvs.iter().map(|(_, v)| *v).collect::<Vec<i8>>();
        let array = Int8Array::from(values);

        let res = avx512_kv_sort(&array, idxs);

        kvs.sort_by(|a, b| a.1.cmp(&b.1));
        assert_eq!(res, kvs);
    }

    #[test]
    fn test_vec_sort_i16_array_avx512() {
        let mut kvs = [4, 3, 2, 1_i16]
            .repeat(2_i32.pow(18) as usize)
            .iter()
            .enumerate()
            .map(|(idx, e)| (idx as u32, *e))
            .collect::<Vec<(u32, i16)>>();

        let idxs = kvs.iter().map(|(k, _)| *k).collect::<Vec<u32>>();
        let values = kvs.iter().map(|(_, v)| *v).collect::<Vec<i16>>();
        let array = Int16Array::from(values);

        let res = avx512_kv_sort(&array, idxs);

        kvs.sort_by(|a, b| a.1.cmp(&b.1));
        assert_eq!(res, kvs);
    }

    #[test]
    fn test_vec_sort_i32_array_avx512() {
        let mut kvs = [i32::max_value(), 3, i32::min_value(), 1_i32]
            .repeat(2_i32.pow(18) as usize)
            .iter()
            .enumerate()
            .map(|(idx, e)| (idx as u32, *e))
            .collect::<Vec<(u32, i32)>>();

        let idxs = kvs.iter().map(|(k, _)| *k).collect::<Vec<u32>>();
        let values = kvs.iter().map(|(_, v)| *v).collect::<Vec<i32>>();
        let array = Int32Array::from(values);

        let res = avx512_kv_sort(&array, idxs);

        kvs.sort_by(|a, b| a.1.cmp(&b.1));
        assert_eq!(res, kvs);
    }

    #[test]
    fn test_vec_sort_u8_array_avx512() {
        let mut kvs = [u8::max_value(), 3, u8::min_value(), 1_u8]
            .repeat(2_i32.pow(18) as usize)
            .iter()
            .enumerate()
            .map(|(idx, e)| (idx as u32, *e))
            .collect::<Vec<(u32, u8)>>();

        let idxs = kvs.iter().map(|(k, _)| *k).collect::<Vec<u32>>();
        let values = kvs.iter().map(|(_, v)| *v).collect::<Vec<u8>>();
        let array = UInt8Array::from(values);

        let res = avx512_kv_sort(&array, idxs);

        kvs.sort_by(|a, b| a.1.cmp(&b.1));
        assert_eq!(res, kvs);
    }

    #[test]
    fn test_vec_sort_u16_array_avx512() {
        let mut kvs = [u16::max_value(), 3, u16::min_value(), 1_u16]
            .repeat(2_i32.pow(18) as usize)
            .iter()
            .enumerate()
            .map(|(idx, e)| (idx as u32, *e))
            .collect::<Vec<(u32, u16)>>();

        let idxs = kvs.iter().map(|(k, _)| *k).collect::<Vec<u32>>();
        let values = kvs.iter().map(|(_, v)| *v).collect::<Vec<u16>>();
        let array = UInt16Array::from(values);

        let res = avx512_kv_sort(&array, idxs);

        kvs.sort_by(|a, b| a.1.cmp(&b.1));
        assert_eq!(res, kvs);
    }
}
