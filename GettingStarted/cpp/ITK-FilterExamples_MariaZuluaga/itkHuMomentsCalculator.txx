#ifndef ITKHUMOMENTSCALCULATOR_TXX
#define ITKHUMOMENTSCALCULATOR_TXX
#include "itkHuMomentsCalculator.h"
#include <itkImageRegionConstIteratorWithIndex.h>

#define X 0
#define Y 1
namespace itk {

template< class TImage >
HuMomentsCalculator< TImage >::HuMomentsCalculator()
{
  m_Valid = false;
  m_Image = NULL;
  m_SpatialObjectMask = NULL;
  m_I1 = NumericTraits< ScalarType >::Zero;
  m_I2 = NumericTraits< ScalarType >::Zero;
  m_I3 = NumericTraits< ScalarType >::Zero;
  m_I4 = NumericTraits< ScalarType >::Zero;
  m_I5 = NumericTraits< ScalarType >::Zero;
  m_I6 = NumericTraits< ScalarType >::Zero;
  m_I7 = NumericTraits< ScalarType >::Zero;
}

template< class TImage >
void
HuMomentsCalculator< TImage >::Compute()
{
  if (m_Valid)
    return;

  if ( !m_Image )
    return;

  typedef typename ImageType::IndexType IndexType;
  MomentValueType mu00 =0;
  MomentValueType firstorder[2] = {0,0};
  ImageRegionConstIteratorWithIndex< ImageType > it( m_Image,
                                                     m_Image->GetRequestedRegion() );
 it.GoToBegin();
 unsigned long count = 0;
  while ( !it.IsAtEnd() )
  {
    double value = it.Value();
    IndexType indexPosition = it.GetIndex();
 //   Point< double, ImageDimension > physicalPosition;
    IndexType physicalPosition = it.GetIndex();

 //   m_Image->TransformIndexToPhysicalPoint(indexPosition, physicalPosition);

    if ( m_SpatialObjectMask.IsNull()
     )//    || m_SpatialObjectMask->IsInside(physicalPosition) )
    {

      mu00 +=value;
      for ( unsigned int i = 0; i < ImageDimension; i++ )
        firstorder[i] += static_cast< double >( physicalPosition[i] ) * value;
    }
    ++it;
    count++;
  }

  if (mu00 == 0)
    return;

  double centroid[ImageDimension];
  for ( unsigned int i = 0; i < ImageDimension; i++ )
    centroid[i] = firstorder[i] / mu00;

  //Compute remaining moments
  MomentValueType mu20 =0, mu02 = 0, mu11 = 0, mu21 = 0,
      mu12 = 0, mu30 = 0, mu03 = 0;
  it.GoToBegin();
  while ( !it.IsAtEnd() )
  {
    double value = it.Value();
    IndexType indexPosition = it.GetIndex();
  //  Point< double, ImageDimension > physicalPosition;
    IndexType physicalPosition = it.GetIndex();
//    m_Image->TransformIndexToPhysicalPoint(indexPosition, physicalPosition);

    if ( m_SpatialObjectMask.IsNull()
       )// || m_SpatialObjectMask->IsInside(physicalPosition) )
    {
      mu20 += pow((physicalPosition[X] - centroid[X]),2)*value;
      mu02 += pow((physicalPosition[Y] - centroid[Y]),2)*value;
      mu11 += (physicalPosition[X] - centroid[X])
          *(physicalPosition[Y] - centroid[Y])*value;
      mu30 += pow((physicalPosition[X] - centroid[X]),3)*value;
      mu03 += pow((physicalPosition[Y] - centroid[Y]),3)*value;
      mu21 += pow((physicalPosition[X] - centroid[X]),2)
          *(physicalPosition[Y] - centroid[Y])*value;
      mu12 += pow((physicalPosition[Y] - centroid[Y]),2)
          *(physicalPosition[X] - centroid[X])*value;
    }
    ++it;
  }

  //Now let's go to nu's
  MomentValueType mu00_sq = mu00*mu00;
  MomentValueType mu00_25 = pow(mu00,2.5);
  MomentValueType nu20 = mu20 / mu00_sq;
  MomentValueType nu02 = mu02 / mu00_sq;
  MomentValueType nu11 = mu11 / mu00_sq;
  MomentValueType nu12 = mu12 / mu00_25;
  MomentValueType nu21 = mu21 / mu00_25;
  MomentValueType nu30 = mu30 / mu00_25;
  MomentValueType nu03 = mu03 / mu00_25;

  //And finally Hu's moments
  MomentValueType nu03_21 = nu03 + nu21;
  MomentValueType nu30_12 = nu30 + nu12;
  MomentValueType nu30_312 = nu30 - 3*nu12;
  MomentValueType nu03_321 = -nu03 + 3*nu21;
  MomentValueType crossed_one = pow((nu30_12),2)-3*pow((nu03_21),2);
  MomentValueType crossed_two = 3*pow((nu30_12),2)-pow((nu03_21),2);

  m_I1 = nu20 + nu02;
  m_I2 = pow(nu20 - nu02,2) + 4*nu11*nu11;
  m_I3 = pow(nu30_312,2)+pow(nu03_321,2);
  m_I4 = pow(nu30_12,2)+pow(nu03_21,2);
  m_I5 = nu30_312*nu30_12*crossed_one
      + (nu03_321*nu03_21*crossed_two);
  m_I6 = (nu20 - nu02)*(pow(nu30_12,2)-pow(nu03_21,2))
      + 4*nu11*nu30_12*nu03_21;
  m_I7 = nu03_321*nu30_12*crossed_one - (nu30_312*nu03_21*crossed_two);
  m_Mass = mu00;
  m_Valid = 1;
}


template< class TInputImage >
void
HuMomentsCalculator< TInputImage >
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Image: " << m_Image.GetPointer() << std::endl;
  os << indent << "Valid: " << m_Valid << std::endl;

}

} // end namespace

#endif //ITKHUMOMENTSCALCULATOR_TXX

