// in order to implement the error correction inside the decoder we need the cases as below:
// 1. The decoder decode each slice and store each CTU of each description in separated buffer.
// 2. After having decoded each slice of the description, the decoder can distinguish which one is corrupted or not
// 3. If the CTU is corrupted of one description, copy the CTU from other description to replace.
// 4. After that the merging process is launched. 
#include "TLibCommon/CommonDef.h"
struct TDecCtx{
    std::vector <Int> LostPOC;
    Bool getMore;
    Bool reRecodedMisMatchSlice;
    Int POC;
};