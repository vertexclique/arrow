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

use crate::buffer::Buffer;

use bitvec::prelude::*;
use bitvec::slice::{ChunksExact};

use std::fmt::Debug;

#[derive(Debug)]
pub struct BufferBitSlice<'a> {
    buffer_data: &'a [u8],
    bit_slice: &'a BitSlice<LocalBits, u8>,
}

impl<'a> BufferBitSlice<'a> {
    pub fn new(buffer_data: &'a [u8]) -> Self {
        let bit_slice = BitSlice::<LocalBits, _>::from_slice(buffer_data).unwrap();

        BufferBitSlice {
            buffer_data,
            bit_slice,
        }
    }

    pub fn view(&self, offset_in_bits: usize, len_in_bits: usize) -> Self {
        Self {
            buffer_data: self.buffer_data,
            bit_slice: &self.bit_slice[offset_in_bits..offset_in_bits + len_in_bits],
        }
    }

    pub fn chunks(&self) -> BufferBitChunksExact {
        let offset_size_in_bits = 8 * std::mem::size_of::<u64>();
        dbg!(offset_size_in_bits);
        BufferBitChunksExact {
            chunks_exact: self.bit_slice.chunks_exact(offset_size_in_bits),
        }
    }

    pub fn into_buffer(&self) -> Buffer {
        Buffer::from(self.bit_slice.as_slice())
    }
}

#[derive(Clone, Debug)]
pub struct BufferBitChunksExact<'a> {
    chunks_exact: ChunksExact<'a, LocalBits, u8>,
}

impl<'a> BufferBitChunksExact<'a> {
    #[inline]
    pub fn remainder_bit_len(&self) -> usize {
        self.chunks_exact.remainder().len()
    }

    #[inline]
    pub fn remainder_bits<T>(&self) -> T
    where
        T: BitMemory,
    {
        let remainder = self.chunks_exact.remainder();
        if remainder.len() == 0 {
            T::default()
        } else {
            self.chunks_exact.remainder().load::<T>()
        }
    }

    #[inline]
    pub fn interpret<T>(self) -> impl Iterator<Item = T> + 'a
    where
        T: BitMemory,
    {
        self.chunks_exact.map(|e| e.load::<T>())
    }

    #[inline]
    pub fn iter(&self) -> &ChunksExact<'a, LocalBits, u8> {
        &self.chunks_exact
    }
}

impl<'a> IntoIterator for BufferBitChunksExact<'a> {
    type Item = &'a BitSlice<LocalBits, u8>;
    // type Item = u64;
    type IntoIter = ChunksExact<'a, LocalBits, u8>;

    fn into_iter(self) -> Self::IntoIter {
        self.chunks_exact
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter_aligned() {
        let input: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7];
        let buffer: Buffer = Buffer::from(input);

        let bit_slice = buffer.bit_slice();
        let result = bit_slice.chunks().interpret().collect::<Vec<u64>>();

        assert_eq!(vec![0x0706050403020100], result);
    }

    #[test]
    fn test_iter_unaligned() {
        let input: &[u8] = &[
            0b00000000, 0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000,
            0b00100000, 0b01000000, 0b11111111,
        ];
        let buffer: Buffer = Buffer::from(input);

        let bit_slice = buffer.bit_slice().view(4, 64);
        let chunks = bit_slice.chunks();

        assert_eq!(0, chunks.remainder_bit_len());
        assert_eq!(0, chunks.remainder_bits::<u64>());

        let result = chunks.interpret().collect::<Vec<u64>>();

        //assert_eq!(vec![0b00010000, 0b00100000, 0b01000000, 0b10000000, 0b00000000, 0b00000001, 0b00000010, 0b11110100], result);
        assert_eq!(
            vec![0b1111010000000010000000010000000010000000010000000010000000010000],
            result
        );
    }

    #[test]
    fn test_iter_unaligned_remainder_1_byte() {
        let input: &[u8] = &[
            0b00000000, 0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000,
            0b00100000, 0b01000000, 0b11111111,
        ];
        let buffer: Buffer = Buffer::from(input);

        let bit_slice = buffer.bit_slice().view(4, 66);
        let chunks = bit_slice.chunks();

        assert_eq!(2, chunks.remainder_bit_len());
        assert_eq!(0b00000011, chunks.remainder_bits::<u64>());

        let result = chunks.interpret().collect::<Vec<u64>>();

        //assert_eq!(vec![0b00010000, 0b00100000, 0b01000000, 0b10000000, 0b00000000, 0b00000001, 0b00000010, 0b11110100], result);
        assert_eq!(
            vec![0b1111010000000010000000010000000010000000010000000010000000010000],
            result
        );
    }
}
