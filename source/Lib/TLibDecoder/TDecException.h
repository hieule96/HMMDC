// make a custom exception class of TDecException for the different case of bitstream error
// and throw it when the bitstream is not valid
//
// The exception class should be derived from std::exception
// we can define only 3 types of error which can happen in the bitstream
// case 1: lost the header and half of the data
// case 2: lost the the half remaining data
// case 3: lost the header and the remaining data
// we can only handle the first one, the other two cases are will be skipped.
#include <exception>
typedef enum AnnexBException
{
    EXCEPTION_PREFIX_ERROR = 0,
    EXCEPTION_TRAILING_ERROR,
    EXCEPTION_TYPE_OTHER
} AnnexBException;


class AnnexeBReadException : public std::exception
{
public:
  AnnexeBReadException(AnnexBException exception) : m_exceptionType(exception) {
    switch (m_exceptionType)
    {
    case EXCEPTION_PREFIX_ERROR:
        m_msg = "[Annexe B]Error at the begin of NAL UNIT slice, a null slice will be given";
        break;
    case EXCEPTION_TRAILING_ERROR:
        m_msg = "[Annexe B]Error at the end of NAL UNIT slice, continue with what the parser has";
        break;
    default:
        m_msg = "[Annexe B]Unhandle exception";
        break;
    }
  }
  AnnexBException getExceptionType() { return m_exceptionType; }
  virtual const char* what() const throw() { return m_msg; }
private:
    const char* m_msg;
    AnnexBException m_exceptionType;
};
