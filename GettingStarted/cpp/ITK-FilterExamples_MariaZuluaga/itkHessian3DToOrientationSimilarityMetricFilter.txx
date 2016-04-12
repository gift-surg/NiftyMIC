#ifndef ITKHESSIANTOORIENTATIONSIMILARITYMETRICFILTER_TXX
#define ITKHESSIANTOORIENTATIONSIMILARITYMETRICFILTER_TXX

#include "itkHessian3DToOrientationSimilarityMetricFilter.h"

#include <itkImageRegionIterator.h>
#include <itkMath.h>
#include <itkImageRegionConstIterator.h>
#include <vnl/vnl_math.h>

namespace itk {

template< typename TPixel >
Hessian3DToOrientationSimilarityMetricFilter< TPixel >
::Hessian3DToOrientationSimilarityMetricFilter()
{
  this->SetNumberOfRequiredInputs(2);
  m_DirectionIndex = 1;
}

template< typename TPixel >
void Hessian3DToOrientationSimilarityMetricFilter< TPixel >
::SetImageOne(const InputImageType* image)
{
  this->SetNthInput(0, const_cast<InputImageType*>(image));
}

template< typename TPixel >
void Hessian3DToOrientationSimilarityMetricFilter< TPixel >
::SetImageTwo(const InputImageType* image)
{
  this->SetNthInput(1, const_cast<InputImageType*>(image));
}



template< typename TPixel >
void Hessian3DToOrientationSimilarityMetricFilter< TPixel >
::GenerateData()
{
  typename OutputImageType::Pointer output = this->GetOutput();

  // walk the region of eigen values
  typename InputImageType::ConstPointer imOne = static_cast< const InputImageType * >
      ( this->ProcessObject::GetInput(0) );
  typename InputImageType::ConstPointer imTwo = static_cast< const InputImageType * >
      ( this->ProcessObject::GetInput(1) );

  ImageRegionConstIterator<InputImageType> itOne, itTwo;
  itOne = ImageRegionConstIterator<InputImageType>(
      imOne, imOne->GetRequestedRegion());
  itTwo = ImageRegionConstIterator<InputImageType>(
      imTwo, imTwo->GetRequestedRegion());
  ImageRegionIterator<OutputImageType> oit;
  this->AllocateOutputs();
  oit = ImageRegionIterator<OutputImageType>(output,
                                             output->GetRequestedRegion());
  //Initialise analyser
  EigenAnalysisType  eig;
  eig.SetDimension( ImageDimension );
  eig.SetOrderEigenMagnitudes( true );
  eig.SetOrderEigenValues( false );
  EigenVectorType eigenMatrixOne, eigenMatrixTwo, tmpMatrix, tmpinvMatrix;
  eigenMatrixOne.Fill(0);
  eigenMatrixTwo.Fill(0);
  EigenValueType eigenValOne, eigenValTwo;
  eigenValOne.Fill(0);
  eigenValTwo.Fill(0);

  oit.GoToBegin();
  itOne.GoToBegin();
  itTwo.GoToBegin();
  while (!oit.IsAtEnd())
  {
    bool failed_1 = false, failed_2 = false;
    SymmetricSecondRankTensor<double, 3> tmpTensor = itOne.Get();
    //ImgOne values
    tmpMatrix[0][0] = tmpTensor[0];
    tmpMatrix[0][1] = tmpTensor[1];
    tmpMatrix[0][2] = tmpTensor[2];
    tmpMatrix[1][0] = tmpTensor[1];
    tmpMatrix[1][1] = tmpTensor[3];
    tmpMatrix[1][2] = tmpTensor[4];
    tmpMatrix[2][0] = tmpTensor[2];
    tmpMatrix[2][1] = tmpTensor[4];
    tmpMatrix[2][2] = tmpTensor[5];

    if (m_DirectionIndex == 3)
    {
      try
      {
        vnl_matrix_fixed< double, 3, 3 > tmpinvvnlmatrix = tmpMatrix.GetInverse();
        tmpinvMatrix = -tmpinvvnlmatrix;
        eig.ComputeEigenValuesAndVectors( tmpinvMatrix, eigenValOne, eigenMatrixOne );
      }
      catch(ExceptionObject e)
      {
        failed_1 = true;
        std::cerr << e.GetDescription() << std::endl;
        std::cout << " Caught an error perhaps due to the determinant" << std::endl;
        eig.ComputeEigenValuesAndVectors( tmpMatrix, eigenValOne, eigenMatrixOne );
        std::cout << " Eigenvalues where: " << eigenValOne[0] << "  " <<
                      eigenValOne[1] << "  " << eigenValOne[2] << "  " << std::endl;
      }
      //Img 2 values
      tmpTensor = itTwo.Get();
      tmpMatrix[0][0] = tmpTensor[0];
      tmpMatrix[0][1] = tmpTensor[1];
      tmpMatrix[0][2] = tmpTensor[2];
      tmpMatrix[1][0] = tmpTensor[1];
      tmpMatrix[1][1] = tmpTensor[3];
      tmpMatrix[1][2] = tmpTensor[4];
      tmpMatrix[2][0] = tmpTensor[2];
      tmpMatrix[2][1] = tmpTensor[4];
      tmpMatrix[2][2] = tmpTensor[5];

      try
      {
        vnl_matrix_fixed< double, 3, 3 > tmpinvvnlmatrix2 = tmpMatrix.GetInverse();
        tmpinvMatrix = -tmpinvvnlmatrix2;
        eig.ComputeEigenValuesAndVectors( tmpinvMatrix, eigenValTwo, eigenMatrixTwo );
      }catch(ExceptionObject e)
      {
        failed_2 = true;
        std::cerr << e.GetDescription() << std::endl;
        std::cout << " Caught an error perhaps due to the determinant" << std::endl;
        eig.ComputeEigenValuesAndVectors( tmpMatrix, eigenValTwo, eigenMatrixTwo );
        std::cout << " Eigenvalues where: " << eigenValTwo[0] << "  " <<
                      eigenValTwo[1] << "  " << eigenValTwo[2] << "  " << std::endl;
      }
    }
    else
    {
      //Img 1 values
       eig.ComputeEigenValuesAndVectors( tmpMatrix, eigenValOne, eigenMatrixOne );
       //Img 2 values
       tmpTensor = itTwo.Get();
       tmpMatrix[0][0] = tmpTensor[0];
       tmpMatrix[0][1] = tmpTensor[1];
       tmpMatrix[0][2] = tmpTensor[2];
       tmpMatrix[1][0] = tmpTensor[1];
       tmpMatrix[1][1] = tmpTensor[3];
       tmpMatrix[1][2] = tmpTensor[4];
       tmpMatrix[2][0] = tmpTensor[2];
       tmpMatrix[2][1] = tmpTensor[4];
       tmpMatrix[2][2] = tmpTensor[5];
       eig.ComputeEigenValuesAndVectors( tmpMatrix, eigenValTwo, eigenMatrixTwo );
    }
    //Order eigenvalues by hand to be sure it works
    unsigned int index1_one, index2_one, index3_one,
        index1_two, index2_two, index3_two;

    this->OrderEigenValuesByMagnitude(eigenValOne,index1_one,index2_one,index3_one);
    this->OrderEigenValuesByMagnitude(eigenValTwo,index1_two,index2_two,index3_two);

    unsigned int index_one, index_two;

    //If diffusion is being used, the largest eigenvalue/vector has to be used to compute angle (3).
    //If hessian is the smallest (1)
    if (m_DirectionIndex == 3)
    {
      index_one = index3_one;
      index_two = index3_two;
    }
    else
    {
      index_one = index1_one;
      index_two = index1_two;
    }

    const double EPSILON = 1e-03;
    bool reject_value = false;
    if (m_DirectionIndex == 3)
    {
      if (eigenValOne[index2_one] < 0 || eigenValOne[index1_one] < 0 ||
          eigenValTwo[index2_two] < 0 || eigenValTwo[index1_two] < 0) //Diffusion rule for rejection
        reject_value = true;
    }
    else  // Hessian rule for rejection
    {
      if ( eigenValOne[index2_one] >= 0.0 ||  eigenValOne[index3_one] >= 0.0 ||
              vnl_math_abs( eigenValOne[index2_one] ) < EPSILON  ||
              vnl_math_abs( eigenValOne[index3_one] ) < EPSILON ||
           eigenValTwo[index2_two] >= 0.0 ||  eigenValTwo[index3_two] >= 0.0 ||
                         vnl_math_abs( eigenValTwo[index2_two] ) < EPSILON  ||
                         vnl_math_abs( eigenValTwo[index3_two] ) < EPSILON)
      {
        reject_value = true;
      }
    }

    if ((m_DirectionIndex == 3 && (failed_1 || failed_2)) || reject_value)
    {
      oit.Set( NumericTraits< OutputPixelType >::Zero );
    }
    else
    {
      OutputPixelType value;
      value = vnl_math_abs((eigenMatrixOne[index_one][0]*eigenMatrixTwo[index_two][0] +
                            eigenMatrixOne[index_one][1]*eigenMatrixTwo[index_two][1] +
                            eigenMatrixOne[index_one][2]*eigenMatrixTwo[index_two][2]) /
                (vcl_sqrt(vnl_math_sqr(eigenMatrixOne[index_one][0])+
                          vnl_math_sqr(eigenMatrixOne[index_one][1])+
                          vnl_math_sqr(eigenMatrixOne[index_one][2])) *
                 vcl_sqrt(vnl_math_sqr(eigenMatrixTwo[index_two][0])+
                          vnl_math_sqr(eigenMatrixTwo[index_two][1])+
                          vnl_math_sqr(eigenMatrixTwo[index_two][2]))));
      oit.Set( value );
    }
    ++oit;
    ++itOne;
    ++itTwo;
  }
}
template< typename TPixel >
void
Hessian3DToOrientationSimilarityMetricFilter< TPixel >
::OrderEigenValuesByMagnitude(EigenValueType eigenVal, unsigned int &indexone,
                                 unsigned int &indextwo, unsigned int &indexthree)
{
  double smallest = vnl_math_abs( eigenVal[0] );
  double Lambda1 = eigenVal[0];
  indexone = 0, indextwo = 0, indexthree = 0;

  for ( unsigned int i=1; i <=2; i++ )
  {
    if ( vnl_math_abs( eigenVal[i] ) < smallest )
    {
      Lambda1 = eigenVal[i];
      smallest = vnl_math_abs( eigenVal[i] );
      indexone = i;
    }
  }
//    // Find the largest eigenvalue
  double largest = vnl_math_abs( eigenVal[0] );
  double Lambda3 = eigenVal[0];

  for ( unsigned int i=1; i <=2; i++ )
  {
    if (  vnl_math_abs( eigenVal[i] ) > largest  )
    {
      Lambda3 = eigenVal[i];
      largest = vnl_math_abs( eigenVal[i] );
      indexthree = i;
    }
  }

  //  find Lambda2 so that |Lambda1| < |Lambda2| < |Lambda3|
 // double Lambda2 = eigenVal[0];

  for ( unsigned int i=0; i <=2; i++ )
  {
    if ( eigenVal[i] != Lambda1 && eigenVal[i] != Lambda3 )
    {
     // Lambda2 = eigenVal[i];
      indextwo =i;
      break;
    }
  }
}

/* ---------------------------------------------------------------------
   PrintSelf method
   --------------------------------------------------------------------- */

template< typename TPixel >
void
Hessian3DToOrientationSimilarityMetricFilter< TPixel >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

}// end namespace

#endif //ITKHESSIANTOORIENTATIONSIMILARITYMETRICFILTER_TXX


