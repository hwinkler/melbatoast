///
/// Given a hypervolume, extract a one-dimensional array from it.
/// The hypervolume is presumably stored in a one dimensional array,
/// with the first dimension indices varying most rapidly.
/// Specify fixed indices for each dimension, save the dimension
/// you want to extract. The function will populate the output
/// parameters offsetOut, lengthOut, and strideOut with the offset,
/// number of elements, and stride between elements of the one
/// dimensional projection.
///

void projection (
                 const int *const dimensions,   /// Array of the lengths of each dimension.
                 const int *const indices,      /// Array of fixed indices for each dimension.
                                                /// One should be -1, meanning unfixed.
                 int numDimensions, /// Length of the dimensions and indices arrays.
                 int *offsetOut,    /// Output: start output position index.
                 int *lengthOut,    /// Output: length of output dimension.
                 int *strideOut);   /// Output: stride between output elements.
