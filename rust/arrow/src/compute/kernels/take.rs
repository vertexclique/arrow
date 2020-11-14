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

//! Defines take kernel for `ArrayRef`

use std::{ops::AddAssign, sync::Arc};

use crate::buffer::{Buffer, MutableBuffer};
use crate::compute::util::take_value_indices_from_list;
use crate::datatypes::*;
use crate::error::{ArrowError, Result};
use crate::util::utils;
use crate::{array::*, buffer::buffer_bin_and};

use num::{ToPrimitive, Zero};
use TimeUnit::*;

/// Take elements from `ArrayRef` by copying the data from `values` at
/// each index in `indices` into a new `ArrayRef`
///
/// For example:
/// ```
/// use std::sync::Arc;
/// use arrow::array::{Array, StringArray, UInt32Array};
/// use arrow::compute::take;
///
/// let values = StringArray::from(vec!["zero", "one", "two"]);
/// let values: Arc<dyn Array> = Arc::new(values);
///
/// // Take items at index 2, and 1:
/// let indices = UInt32Array::from(vec![2, 1]);
/// let taken = take(&values, &indices, None).unwrap();
/// let taken = taken.as_any().downcast_ref::<StringArray>().unwrap();
///
/// assert_eq!(*taken, StringArray::from(vec!["two", "one"]));
/// ```
///
/// Supports:
///  * null indices, returning a null value for the index
///  * checking for overflowing indices
pub fn take(
    values: &ArrayRef,
    indices: &UInt32Array,
    options: Option<TakeOptions>,
) -> Result<ArrayRef> {
    take_impl::<UInt32Type>(values, indices, options)
}

fn take_impl<IndexType>(
    values: &ArrayRef,
    indices: &PrimitiveArray<IndexType>,
    options: Option<TakeOptions>,
) -> Result<ArrayRef>
where
    IndexType: ArrowNumericType,
    IndexType::Native: ToPrimitive,
{
    let options = options.unwrap_or_default();
    if options.check_bounds {
        let len = values.len();
        for i in 0..indices.len() {
            if indices.is_valid(i) {
                let ix = ToPrimitive::to_usize(&indices.value(i)).ok_or_else(|| {
                    ArrowError::ComputeError("Cast to usize failed".to_string())
                })?;
                if ix >= len {
                    return Err(ArrowError::ComputeError(
                    format!("Array index out of bounds, cannot get item at index {} from {} entries", ix, len))
                );
                }
            }
        }
    }
    match values.data_type() {
        DataType::Boolean => take_boolean(values, indices),
        DataType::Int8 => take_primitive::<Int8Type, _>(values, indices),
        DataType::Int16 => take_primitive::<Int16Type, _>(values, indices),
        DataType::Int32 => take_primitive::<Int32Type, _>(values, indices),
        DataType::Int64 => take_primitive::<Int64Type, _>(values, indices),
        DataType::UInt8 => take_primitive::<UInt8Type, _>(values, indices),
        DataType::UInt16 => take_primitive::<UInt16Type, _>(values, indices),
        DataType::UInt32 => take_primitive::<UInt32Type, _>(values, indices),
        DataType::UInt64 => take_primitive::<UInt64Type, _>(values, indices),
        DataType::Float32 => take_primitive::<Float32Type, _>(values, indices),
        DataType::Float64 => take_primitive::<Float64Type, _>(values, indices),
        DataType::Date32(_) => take_primitive::<Date32Type, _>(values, indices),
        DataType::Date64(_) => take_primitive::<Date64Type, _>(values, indices),
        DataType::Time32(Second) => {
            take_primitive::<Time32SecondType, _>(values, indices)
        }
        DataType::Time32(Millisecond) => {
            take_primitive::<Time32MillisecondType, _>(values, indices)
        }
        DataType::Time64(Microsecond) => {
            take_primitive::<Time64MicrosecondType, _>(values, indices)
        }
        DataType::Time64(Nanosecond) => {
            take_primitive::<Time64NanosecondType, _>(values, indices)
        }
        DataType::Timestamp(Second, _) => {
            take_primitive::<TimestampSecondType, _>(values, indices)
        }
        DataType::Timestamp(Millisecond, _) => {
            take_primitive::<TimestampMillisecondType, _>(values, indices)
        }
        DataType::Timestamp(Microsecond, _) => {
            take_primitive::<TimestampMicrosecondType, _>(values, indices)
        }
        DataType::Timestamp(Nanosecond, _) => {
            take_primitive::<TimestampNanosecondType, _>(values, indices)
        }
        DataType::Interval(IntervalUnit::YearMonth) => {
            take_primitive::<IntervalYearMonthType, _>(values, indices)
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            take_primitive::<IntervalDayTimeType, _>(values, indices)
        }
        DataType::Duration(TimeUnit::Second) => {
            take_primitive::<DurationSecondType, _>(values, indices)
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            take_primitive::<DurationMillisecondType, _>(values, indices)
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            take_primitive::<DurationMicrosecondType, _>(values, indices)
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            take_primitive::<DurationNanosecondType, _>(values, indices)
        }
        DataType::Utf8 => take_string::<i32, _>(values, indices),
        DataType::LargeUtf8 => take_string::<i64, _>(values, indices),
        DataType::List(_) => take_list::<_, Int32Type>(values, indices),
        DataType::LargeList(_) => take_list::<_, Int64Type>(values, indices),
        DataType::Struct(fields) => {
            let struct_: &StructArray =
                values.as_any().downcast_ref::<StructArray>().unwrap();
            let arrays: Result<Vec<ArrayRef>> = struct_
                .columns()
                .iter()
                .map(|a| take_impl(a, indices, Some(options.clone())))
                .collect();
            let arrays = arrays?;
            let pairs: Vec<(Field, ArrayRef)> =
                fields.clone().into_iter().zip(arrays).collect();
            Ok(Arc::new(StructArray::from(pairs)) as ArrayRef)
        }
        DataType::Dictionary(key_type, _) => match key_type.as_ref() {
            DataType::Int8 => take_dict::<Int8Type, _>(values, indices),
            DataType::Int16 => take_dict::<Int16Type, _>(values, indices),
            DataType::Int32 => take_dict::<Int32Type, _>(values, indices),
            DataType::Int64 => take_dict::<Int64Type, _>(values, indices),
            DataType::UInt8 => take_dict::<UInt8Type, _>(values, indices),
            DataType::UInt16 => take_dict::<UInt16Type, _>(values, indices),
            DataType::UInt32 => take_dict::<UInt32Type, _>(values, indices),
            DataType::UInt64 => take_dict::<UInt64Type, _>(values, indices),
            t => unimplemented!("Take not supported for dictionary key type {:?}", t),
        },
        t => unimplemented!("Take not supported for data type {:?}", t),
    }
}

/// Options that define how `take` should behave
#[derive(Clone, Debug)]
pub struct TakeOptions {
    /// Perform bounds check before taking indices from values.
    /// If enabled, an `ArrowError` is returned if the indices are out of bounds.
    /// If not enabled, and indices exceed bounds, the kernel will panic.
    pub check_bounds: bool,
}

impl Default for TakeOptions {
    fn default() -> Self {
        Self {
            check_bounds: false,
        }
    }
}

/// `take` implementation for all primitive arrays except boolean
///
/// This checks if an `indices` slot is populated, and gets the value from `values`
///  as the populated index.
/// If the `indices` slot is null, a null value is returned.
/// For example, given:
///     values:  [1, 2, 3, null, 5]
///     indices: [0, null, 4, 3]
/// The result is: [1 (slot 0), null (null slot), 5 (slot 4), null (slot 3)]
fn take_primitive<T, I>(
    values: &ArrayRef,
    indices: &PrimitiveArray<I>,
) -> Result<ArrayRef>
where
    T: ArrowPrimitiveType,
    I: ArrowNumericType,
    I::Native: ToPrimitive,
{
    let data_len = indices.len();

    let array = values.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();

    let num_bytes = utils::ceil(data_len, 8);
    let mut null_buf = MutableBuffer::new(num_bytes).with_bitset(num_bytes, true);

    // This iteration is implemented with a while loop, rather than a
    // map()/collect(), since the while loop performs better in the benchmarks.
    let mut new_values: Vec<T::Native> = Vec::with_capacity(data_len);
    let mut i = 0;
    while i < data_len {
        let index = ToPrimitive::to_usize(&indices.value(i)).ok_or_else(|| {
            ArrowError::ComputeError("Cast to usize failed".to_string())
        })?;

        if array.is_null(index) {
            null_buf.unset_bit(i);
        }

        new_values.push(array.value(index));

        i += 1;
    }

    let nulls = match indices.data_ref().null_buffer() {
        Some(buffer) => buffer_bin_and(buffer, 0, &null_buf.freeze(), 0, indices.len()),
        None => null_buf.freeze(),
    };

    let data = ArrayData::new(
        T::DATA_TYPE,
        indices.len(),
        None,
        Some(nulls),
        0,
        vec![Buffer::from(new_values.to_byte_slice())],
        vec![],
    );
    Ok(Arc::new(PrimitiveArray::<T>::from(Arc::new(data))))
}

/// `take` implementation for boolean arrays
fn take_boolean<IndexType>(
    values: &ArrayRef,
    indices: &PrimitiveArray<IndexType>,
) -> Result<ArrayRef>
where
    IndexType: ArrowNumericType,
    IndexType::Native: ToPrimitive,
{
    let data_len = indices.len();

    let array = values.as_any().downcast_ref::<BooleanArray>().unwrap();

    let num_byte = utils::ceil(data_len, 8);
    let mut null_buf = MutableBuffer::new(num_byte).with_bitset(num_byte, true);
    let mut val_buf = MutableBuffer::new(num_byte).with_bitset(num_byte, false);

    (0..data_len).try_for_each::<_, Result<()>>(|i| {
        let index = ToPrimitive::to_usize(&indices.value(i)).ok_or_else(|| {
            ArrowError::ComputeError("Cast to usize failed".to_string())
        })?;

        if array.is_null(index) {
            null_buf.unset_bit(i);
        } else if array.value(index) {
            val_buf.set_bit(i);
        }

        Ok(())
    })?;

    let nulls = match indices.data_ref().null_buffer() {
        Some(buffer) => buffer_bin_and(buffer, 0, &null_buf.freeze(), 0, indices.len()),
        None => null_buf.freeze(),
    };

    let data = ArrayData::new(
        DataType::Boolean,
        indices.len(),
        None,
        Some(nulls),
        0,
        vec![val_buf.freeze()],
        vec![],
    );
    Ok(Arc::new(BooleanArray::from(Arc::new(data))))
}

/// `take` implementation for string arrays
fn take_string<OffsetSize, IndexType>(
    values: &ArrayRef,
    indices: &PrimitiveArray<IndexType>,
) -> Result<ArrayRef>
where
    OffsetSize: Zero + AddAssign + StringOffsetSizeTrait,
    IndexType: ArrowNumericType,
    IndexType::Native: ToPrimitive,
{
    let data_len = indices.len();

    let array = values
        .as_any()
        .downcast_ref::<GenericStringArray<OffsetSize>>()
        .unwrap();

    let num_bytes = utils::ceil(data_len, 8);
    let mut null_buf = MutableBuffer::new(num_bytes).with_bitset(num_bytes, true);

    let mut offsets = Vec::with_capacity(data_len + 1);
    let mut values = Vec::with_capacity(data_len);
    let mut length_so_far = OffsetSize::zero();

    offsets.push(length_so_far);
    for i in 0..data_len {
        let index = ToPrimitive::to_usize(&indices.value(i)).ok_or_else(|| {
            ArrowError::ComputeError("Cast to usize failed".to_string())
        })?;

        if array.is_valid(index) && indices.is_valid(i) {
            let s = array.value(index);

            length_so_far += OffsetSize::from_usize(s.len()).unwrap();
            values.extend_from_slice(s.as_bytes());
        } else {
            // set null bit
            null_buf.unset_bit(i);
        }
        offsets.push(length_so_far);
    }

    let nulls = match indices.data_ref().null_buffer() {
        Some(buffer) => buffer_bin_and(buffer, 0, &null_buf.freeze(), 0, data_len),
        None => null_buf.freeze(),
    };

    let data = ArrayData::builder(<OffsetSize as StringOffsetSizeTrait>::DATA_TYPE)
        .len(data_len)
        .null_bit_buffer(nulls)
        .add_buffer(Buffer::from(offsets.to_byte_slice()))
        .add_buffer(Buffer::from(&values[..]))
        .build();
    Ok(Arc::new(GenericStringArray::<OffsetSize>::from(data)))
}

/// `take` implementation for list arrays
///
/// Calculates the index and indexed offset for the inner array,
/// applying `take` on the inner array, then reconstructing a list array
/// with the indexed offsets
fn take_list<IndexType, OffsetType>(
    values: &ArrayRef,
    indices: &PrimitiveArray<IndexType>,
) -> Result<ArrayRef>
where
    IndexType: ArrowNumericType,
    IndexType::Native: ToPrimitive,
    OffsetType: ArrowNumericType,
    OffsetType::Native: ToPrimitive + OffsetSizeTrait,
    PrimitiveArray<OffsetType>: From<Vec<Option<OffsetType::Native>>>,
{
    // TODO: Some optimizations can be done here such as if it is
    // taking the whole list or a contiguous sublist
    let list = values
        .as_any()
        .downcast_ref::<GenericListArray<OffsetType::Native>>()
        .unwrap();

    let (list_indices, offsets) =
        take_value_indices_from_list::<IndexType, OffsetType>(values, indices)?;

    let taken = take_impl::<OffsetType>(&list.values(), &list_indices, None)?;
    // determine null count and null buffer, which are a function of `values` and `indices`
    let num_bytes = utils::ceil(indices.len(), 8);
    let mut null_buf = MutableBuffer::new(num_bytes).with_bitset(num_bytes, true);
    {
        offsets[..].windows(2).enumerate().for_each(
            |(i, window): (usize, &[OffsetType::Native])| {
                if window[0] == window[1] {
                    // offsets are equal, slot is null
                    null_buf.unset_bit(i);
                }
            },
        );
    }
    let value_offsets = Buffer::from(offsets[..].to_byte_slice());
    // create a new list with taken data and computed null information
    let list_data = ArrayDataBuilder::new(list.data_type().clone())
        .len(indices.len())
        .null_count(null_buf.count_zeros())
        .null_bit_buffer(null_buf.freeze())
        .offset(0)
        .add_child_data(taken.data())
        .add_buffer(value_offsets)
        .build();
    let list_array =
        Arc::new(GenericListArray::<OffsetType::Native>::from(list_data)) as ArrayRef;
    Ok(list_array)
}

/// `take` implementation for dictionary arrays
///
/// applies `take` to the keys of the dictionary array and returns a new dictionary array
/// with the same dictionary values and reordered keys
fn take_dict<T, I>(values: &ArrayRef, indices: &PrimitiveArray<I>) -> Result<ArrayRef>
where
    T: ArrowPrimitiveType,
    I: ArrowNumericType,
    I::Native: ToPrimitive,
{
    let dict = values
        .as_any()
        .downcast_ref::<DictionaryArray<T>>()
        .unwrap();
    let keys: ArrayRef = Arc::new(dict.keys_array());
    let new_keys = take_primitive::<T, I>(&keys, indices)?;
    let new_keys_data = new_keys.data_ref();

    let data = Arc::new(ArrayData::new(
        dict.data_type().clone(),
        new_keys.len(),
        Some(new_keys_data.null_count()),
        new_keys_data.null_buffer().cloned(),
        0,
        new_keys_data.buffers().to_vec(),
        dict.data().child_data().to_vec(),
    ));

    Ok(Arc::new(DictionaryArray::<T>::from(data)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::bit_slice_iterator::*;

    fn test_take_primitive_arrays<T>(
        data: Vec<Option<T::Native>>,
        index: &UInt32Array,
        options: Option<TakeOptions>,
        expected_data: Vec<Option<T::Native>>,
    ) where
        T: ArrowPrimitiveType,
        PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
    {
        let output = PrimitiveArray::<T>::from(data);
        let expected = Arc::new(PrimitiveArray::<T>::from(expected_data)) as ArrayRef;
        let output = take(&(Arc::new(output) as ArrayRef), index, options).unwrap();
        assert_eq!(&output, &expected)
    }

    fn test_take_impl_primitive_arrays<T, I>(
        data: Vec<Option<T::Native>>,
        index: &PrimitiveArray<I>,
        options: Option<TakeOptions>,
        expected_data: Vec<Option<T::Native>>,
    ) where
        T: ArrowPrimitiveType,
        PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
        I: ArrowNumericType,
        I::Native: ToPrimitive,
    {
        let output = PrimitiveArray::<T>::from(data);
        let expected = PrimitiveArray::<T>::from(expected_data);
        let output = take_impl(&(Arc::new(output) as ArrayRef), index, options).unwrap();
        let output = output.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
        assert_eq!(output, &expected)
    }

    // create a simple struct for testing purposes
    fn create_test_struct() -> ArrayRef {
        let boolean_data = BooleanArray::from(vec![true, false, false, true]).data();
        let int_data = Int32Array::from(vec![42, 28, 19, 31]).data();
        let mut field_types = vec![];
        field_types.push(Field::new("a", DataType::Boolean, true));
        field_types.push(Field::new("b", DataType::Int32, true));
        let struct_array_data = ArrayData::builder(DataType::Struct(field_types))
            .len(4)
            .null_count(0)
            .add_child_data(boolean_data)
            .add_child_data(int_data)
            .build();
        let struct_array = StructArray::from(struct_array_data);
        Arc::new(struct_array) as ArrayRef
    }

    #[test]
    fn test_take_primitive() {
        let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(3), Some(2)]);

        // int8
        test_take_primitive_arrays::<Int8Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // int16
        test_take_primitive_arrays::<Int16Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // int32
        test_take_primitive_arrays::<Int32Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // int64
        test_take_primitive_arrays::<Int64Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // uint8
        test_take_primitive_arrays::<UInt8Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // uint16
        test_take_primitive_arrays::<UInt16Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // uint32
        test_take_primitive_arrays::<UInt32Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // int64
        test_take_primitive_arrays::<Int64Type>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // interval_year_month
        test_take_primitive_arrays::<IntervalYearMonthType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // interval_day_time
        test_take_primitive_arrays::<IntervalDayTimeType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // duration_second
        test_take_primitive_arrays::<DurationSecondType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // duration_millisecond
        test_take_primitive_arrays::<DurationMillisecondType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // duration_microsecond
        test_take_primitive_arrays::<DurationMicrosecondType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // duration_nanosecond
        test_take_primitive_arrays::<DurationNanosecondType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // float32
        test_take_primitive_arrays::<Float32Type>(
            vec![Some(0.0), None, Some(2.21), Some(-3.1), None],
            &index,
            None,
            vec![Some(-3.1), None, None, Some(-3.1), Some(2.21)],
        );

        // float64
        test_take_primitive_arrays::<Float64Type>(
            vec![Some(0.0), None, Some(2.21), Some(-3.1), None],
            &index,
            None,
            vec![Some(-3.1), None, None, Some(-3.1), Some(2.21)],
        );
    }

    #[test]
    fn test_take_impl_primitive_with_int64_indices() {
        let index = Int64Array::from(vec![Some(3), None, Some(1), Some(3), Some(2)]);

        // int16
        test_take_impl_primitive_arrays::<Int16Type, Int64Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // int64
        test_take_impl_primitive_arrays::<Int64Type, Int64Type>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // uint64
        test_take_impl_primitive_arrays::<UInt64Type, Int64Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // duration_millisecond
        test_take_impl_primitive_arrays::<DurationMillisecondType, Int64Type>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // float32
        test_take_impl_primitive_arrays::<Float32Type, Int64Type>(
            vec![Some(0.0), None, Some(2.21), Some(-3.1), None],
            &index,
            None,
            vec![Some(-3.1), None, None, Some(-3.1), Some(2.21)],
        );
    }

    #[test]
    fn test_take_impl_primitive_with_uint8_indices() {
        let index = UInt8Array::from(vec![Some(3), None, Some(1), Some(3), Some(2)]);

        // int16
        test_take_impl_primitive_arrays::<Int16Type, UInt8Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // duration_millisecond
        test_take_impl_primitive_arrays::<DurationMillisecondType, UInt8Type>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // float32
        test_take_impl_primitive_arrays::<Float32Type, UInt8Type>(
            vec![Some(0.0), None, Some(2.21), Some(-3.1), None],
            &index,
            None,
            vec![Some(-3.1), None, None, Some(-3.1), Some(2.21)],
        );
    }

    #[test]
    fn test_take_primitive_bool() {
        let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(3), Some(2)]);
        // boolean
        test_take_primitive_arrays::<BooleanType>(
            vec![Some(false), None, Some(true), Some(false), None],
            &index,
            None,
            vec![Some(false), None, None, Some(false), Some(true)],
        );
    }

    fn _test_take_string<'a, K: 'static>()
    where
        K: Array + PartialEq + From<Vec<Option<&'a str>>>,
    {
        let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(3), Some(4)]);

        let array = K::from(vec![
            Some("one"),
            None,
            Some("three"),
            Some("four"),
            Some("five"),
        ]);
        let array = Arc::new(array) as ArrayRef;

        let actual = take(&array, &index, None).unwrap();
        assert_eq!(actual.len(), index.len());

        let actual = actual.as_any().downcast_ref::<K>().unwrap();

        let expected =
            K::from(vec![Some("four"), None, None, Some("four"), Some("five")]);

        assert_eq!(actual, &expected);
    }

    #[test]
    fn test_take_string() {
        _test_take_string::<StringArray>()
    }

    #[test]
    fn test_take_large_string() {
        _test_take_string::<LargeStringArray>()
    }

    macro_rules! test_take_list {
        ($offset_type:ty, $list_data_type:ident, $list_array_type:ident) => {{
            // Construct a value array, [[0,0,0], [-1,-2,-1], [2,3]]
            let value_data = Int32Array::from(vec![0, 0, 0, -1, -2, -1, 2, 3]).data();
            // Construct offsets
            let value_offsets: [$offset_type; 4] = [0, 3, 6, 8];
            let value_offsets = Buffer::from(&value_offsets.to_byte_slice());
            // Construct a list array from the above two
            let list_data_type = DataType::$list_data_type(Box::new(Field::new(
                "item",
                DataType::Int32,
                false,
            )));
            let list_data = ArrayData::builder(list_data_type.clone())
                .len(3)
                .add_buffer(value_offsets)
                .add_child_data(value_data)
                .build();
            let list_array = Arc::new($list_array_type::from(list_data)) as ArrayRef;

            // index returns: [[2,3], null, [-1,-2,-1], [2,3], [0,0,0]]
            let index = UInt32Array::from(vec![Some(2), None, Some(1), Some(2), Some(0)]);

            let a = take(&list_array, &index, None).unwrap();
            let a: &$list_array_type =
                a.as_any().downcast_ref::<$list_array_type>().unwrap();

            // construct a value array with expected results:
            // [[2,3], null, [-1,-2,-1], [2,3], [0,0,0]]
            let expected_data = Int32Array::from(vec![
                Some(2),
                Some(3),
                Some(-1),
                Some(-2),
                Some(-1),
                Some(2),
                Some(3),
                Some(0),
                Some(0),
                Some(0),
            ])
            .data();
            // construct offsets
            let expected_offsets: [$offset_type; 6] = [0, 2, 2, 5, 7, 10];
            let expected_offsets = Buffer::from(&expected_offsets.to_byte_slice());
            // construct list array from the two
            let expected_list_data = ArrayData::builder(list_data_type)
                .len(5)
                .null_count(1)
                // null buffer remains the same as only the indices have nulls
                .null_bit_buffer(
                    index.data().null_bitmap().as_ref().unwrap().bits.clone(),
                )
                .add_buffer(expected_offsets)
                .add_child_data(expected_data)
                .build();
            let expected_list_array = $list_array_type::from(expected_list_data);

            assert_eq!(a, &expected_list_array);
        }};
    }

    macro_rules! test_take_list_with_value_nulls {
        ($offset_type:ty, $list_data_type:ident, $list_array_type:ident) => {{
            // Construct a value array, [[0,null,0], [-1,-2,3], [null], [5,null]]
            let value_data = Int32Array::from(vec![
                Some(0),
                None,
                Some(0),
                Some(-1),
                Some(-2),
                Some(3),
                None,
                Some(5),
                None,
            ])
            .data();
            // Construct offsets
            let value_offsets: [$offset_type; 5] = [0, 3, 6, 7, 9];
            let value_offsets = Buffer::from(&value_offsets.to_byte_slice());
            // Construct a list array from the above two
            let list_data_type = DataType::$list_data_type(Box::new(Field::new(
                "item",
                DataType::Int32,
                false,
            )));
            let list_data = ArrayData::builder(list_data_type.clone())
                .len(4)
                .add_buffer(value_offsets)
                .null_count(0)
                .null_bit_buffer(Buffer::from([0b10111101, 0b00000000]))
                .add_child_data(value_data)
                .build();
            let list_array = Arc::new($list_array_type::from(list_data)) as ArrayRef;

            // index returns: [[null], null, [-1,-2,3], [2,null], [0,null,0]]
            let index = UInt32Array::from(vec![Some(2), None, Some(1), Some(3), Some(0)]);

            let a = take(&list_array, &index, None).unwrap();
            let a: &$list_array_type =
                a.as_any().downcast_ref::<$list_array_type>().unwrap();

            // construct a value array with expected results:
            // [[null], null, [-1,-2,3], [5,null], [0,null,0]]
            let expected_data = Int32Array::from(vec![
                None,
                Some(-1),
                Some(-2),
                Some(3),
                Some(5),
                None,
                Some(0),
                None,
                Some(0),
            ])
            .data();
            // construct offsets
            let expected_offsets: [$offset_type; 6] = [0, 1, 1, 4, 6, 9];
            let expected_offsets = Buffer::from(&expected_offsets.to_byte_slice());
            // construct list array from the two
            let expected_list_data = ArrayData::builder(list_data_type)
                .len(5)
                .null_count(1)
                // null buffer remains the same as only the indices have nulls
                .null_bit_buffer(
                    index.data().null_bitmap().as_ref().unwrap().bits.clone(),
                )
                .add_buffer(expected_offsets)
                .add_child_data(expected_data)
                .build();
            let expected_list_array = $list_array_type::from(expected_list_data);

            assert_eq!(a, &expected_list_array);
        }};
    }

    macro_rules! test_take_list_with_nulls {
        ($offset_type:ty, $list_data_type:ident, $list_array_type:ident) => {{
            // Construct a value array, [[0,null,0], [-1,-2,3], null, [5,null]]
            let value_data = Int32Array::from(vec![
                Some(0),
                None,
                Some(0),
                Some(-1),
                Some(-2),
                Some(3),
                Some(5),
                None,
            ])
            .data();
            // Construct offsets
            let value_offsets: [$offset_type; 5] = [0, 3, 6, 6, 8];
            let value_offsets = Buffer::from(&value_offsets.to_byte_slice());
            // Construct a list array from the above two
            let list_data_type = DataType::$list_data_type(Box::new(Field::new(
                "item",
                DataType::Int32,
                false,
            )));
            let list_data = ArrayData::builder(list_data_type.clone())
                .len(4)
                .add_buffer(value_offsets)
                .null_count(1)
                .null_bit_buffer(Buffer::from([0b01111101]))
                .add_child_data(value_data)
                .build();
            let list_array = Arc::new($list_array_type::from(list_data)) as ArrayRef;

            // index returns: [null, null, [-1,-2,3], [5,null], [0,null,0]]
            let index = UInt32Array::from(vec![Some(2), None, Some(1), Some(3), Some(0)]);

            let a = take(&list_array, &index, None).unwrap();
            let a: &$list_array_type =
                a.as_any().downcast_ref::<$list_array_type>().unwrap();

            // construct a value array with expected results:
            // [null, null, [-1,-2,3], [5,null], [0,null,0]]
            let expected_data = Int32Array::from(vec![
                Some(-1),
                Some(-2),
                Some(3),
                Some(5),
                None,
                Some(0),
                None,
                Some(0),
            ])
            .data();
            // construct offsets
            let expected_offsets: [$offset_type; 6] = [0, 0, 0, 3, 5, 8];
            let expected_offsets = Buffer::from(&expected_offsets.to_byte_slice());
            // construct list array from the two
            let mut null_bits: [u8; 1] = [0; 1];
            let mut null_bit_view = BufferBitSliceMut::new(&mut null_bits);
            null_bit_view.set_bit(2, true);
            null_bit_view.set_bit(3, true);
            null_bit_view.set_bit(4, true);
            let expected_list_data = ArrayData::builder(list_data_type)
                .len(5)
                .null_count(2)
                // null buffer must be recalculated as both values and indices have nulls
                .null_bit_buffer(Buffer::from(null_bits))
                .add_buffer(expected_offsets)
                .add_child_data(expected_data)
                .build();
            let expected_list_array = $list_array_type::from(expected_list_data);

            assert_eq!(a, &expected_list_array);
        }};
    }

    #[test]
    fn test_take_list() {
        test_take_list!(i32, List, ListArray);
    }

    #[test]
    fn test_take_large_list() {
        test_take_list!(i64, LargeList, LargeListArray);
    }

    #[test]
    fn test_take_list_with_value_nulls() {
        test_take_list_with_value_nulls!(i32, List, ListArray);
    }

    #[test]
    fn test_take_large_list_with_value_nulls() {
        test_take_list_with_value_nulls!(i64, LargeList, LargeListArray);
    }

    #[test]
    fn test_test_take_list_with_nulls() {
        test_take_list_with_nulls!(i32, List, ListArray);
    }

    #[test]
    fn test_test_take_large_list_with_nulls() {
        test_take_list_with_nulls!(i64, LargeList, LargeListArray);
    }

    #[test]
    #[should_panic(expected = "index out of bounds: the len is 4 but the index is 1000")]
    fn test_take_list_out_of_bounds() {
        // Construct a value array, [[0,0,0], [-1,-2,-1], [2,3]]
        let value_data = Int32Array::from(vec![0, 0, 0, -1, -2, -1, 2, 3]).data();
        // Construct offsets
        let value_offsets = Buffer::from(&[0, 3, 6, 8].to_byte_slice());
        // Construct a list array from the above two
        let list_data_type =
            DataType::List(Box::new(Field::new("item", DataType::Int32, false)));
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(3)
            .add_buffer(value_offsets)
            .add_child_data(value_data)
            .build();
        let list_array = Arc::new(ListArray::from(list_data)) as ArrayRef;

        let index = UInt32Array::from(vec![1000]);

        // A panic is expected here since we have not supplied the check_bounds
        // option.
        take(&list_array, &index, None).unwrap();
    }

    #[test]
    fn test_take_struct() {
        let array = create_test_struct();

        let index = UInt32Array::from(vec![0, 3, 1, 0, 2]);
        let a = take(&array, &index, None).unwrap();
        let a: &StructArray = a.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(index.len(), a.len());
        assert_eq!(0, a.null_count());

        let expected_bool_data =
            BooleanArray::from(vec![true, true, false, true, false]).data();
        let expected_int_data = Int32Array::from(vec![42, 31, 28, 42, 19]).data();
        let mut field_types = vec![];
        field_types.push(Field::new("a", DataType::Boolean, true));
        field_types.push(Field::new("b", DataType::Int32, true));
        let struct_array_data = ArrayData::builder(DataType::Struct(field_types))
            .len(5)
            .null_count(0)
            .add_child_data(expected_bool_data)
            .add_child_data(expected_int_data)
            .build();
        let struct_array = StructArray::from(struct_array_data);

        assert_eq!(a, &struct_array);
    }

    #[test]
    fn test_take_struct_with_nulls() {
        let array = create_test_struct();

        let index = UInt32Array::from(vec![None, Some(3), Some(1), None, Some(0)]);
        let a = take(&array, &index, None).unwrap();
        let a: &StructArray = a.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(index.len(), a.len());
        assert_eq!(0, a.null_count());

        let expected_bool_data =
            BooleanArray::from(vec![None, Some(true), Some(false), None, Some(true)])
                .data();
        let expected_int_data =
            Int32Array::from(vec![None, Some(31), Some(28), None, Some(42)]).data();

        let mut field_types = vec![];
        field_types.push(Field::new("a", DataType::Boolean, true));
        field_types.push(Field::new("b", DataType::Int32, true));
        let struct_array_data = ArrayData::builder(DataType::Struct(field_types))
            .len(5)
            // TODO: see https://issues.apache.org/jira/browse/ARROW-5408 for why count != 2
            .null_count(0)
            .add_child_data(expected_bool_data)
            .add_child_data(expected_int_data)
            .build();
        let struct_array = StructArray::from(struct_array_data);
        assert_eq!(a, &struct_array);
    }

    #[test]
    #[should_panic(
        expected = "Array index out of bounds, cannot get item at index 6 from 5 entries"
    )]
    fn test_take_out_of_bounds() {
        let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(3), Some(6)]);
        let take_opt = TakeOptions { check_bounds: true };

        // int64
        test_take_primitive_arrays::<Int64Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            Some(take_opt),
            vec![None],
        );
    }

    #[test]
    fn test_take_dict() {
        let keys_builder = Int16Builder::new(8);
        let values_builder = StringBuilder::new(4);

        let mut dict_builder = StringDictionaryBuilder::new(keys_builder, values_builder);

        dict_builder.append("foo").unwrap();
        dict_builder.append("bar").unwrap();
        dict_builder.append("").unwrap();
        dict_builder.append_null().unwrap();
        dict_builder.append("foo").unwrap();
        dict_builder.append("bar").unwrap();
        dict_builder.append("bar").unwrap();
        dict_builder.append("foo").unwrap();

        let array = dict_builder.finish();
        let dict_values = array.values().clone();
        let dict_values = dict_values.as_any().downcast_ref::<StringArray>().unwrap();
        let array: Arc<dyn Array> = Arc::new(array);

        let indices = UInt32Array::from(vec![
            Some(0), // first "foo"
            Some(7), // last "foo"
            None,    // null index should return null
            Some(5), // second "bar"
            Some(6), // another "bar"
            Some(2), // empty string
            Some(3), // input is null at this index
        ]);

        let result = take(&array, &indices, None).unwrap();
        let result = result
            .as_any()
            .downcast_ref::<DictionaryArray<Int16Type>>()
            .unwrap();

        let result_values: StringArray = result.values().data().into();

        // dictionary values should stay the same
        let expected_values = StringArray::from(vec!["foo", "bar", ""]);
        assert_eq!(&expected_values, dict_values);
        assert_eq!(&expected_values, &result_values);

        let expected_keys = Int16Array::from(vec![
            Some(0),
            Some(0),
            None,
            Some(1),
            Some(1),
            Some(2),
            None,
        ]);
        assert_eq!(result.keys(), &expected_keys);
    }
}
