
pub trait ValueAccessors<V> {
    /// Returns value at index `i`.
    fn value(&self, i: usize) -> V;
}