#ifndef ITKHUMOMENTSCALCULATOR_H
#define ITKHUMOMENTSCALCULATOR_H

#include <itkImage.h>
#include <itkSpatialObject.h>

namespace itk {


template< class TImage >
class ITK_EXPORT HuMomentsCalculator : public Object
{
public:
  /** Standard class typedefs. */
  typedef HuMomentsCalculator< TImage > Self;
  typedef Object                           Superclass;
  typedef SmartPointer< Self >             Pointer;
  typedef SmartPointer< const Self >       ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageMomentsCalculator, Object);

  /** Extract the dimension of the image. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TImage::ImageDimension);

  /** Standard scalar type within this class. */
  typedef double ScalarType;

  /** Standard vector type within this class. */
  typedef Vector< ScalarType, itkGetStaticConstMacro(ImageDimension) > VectorType;

  /** Spatial Object type within this class. */
  typedef SpatialObject< itkGetStaticConstMacro(ImageDimension) > SpatialObjectType;

  /** Spatial Object member types used within this class. */
  typedef typename SpatialObjectType::Pointer      SpatialObjectPointer;
  typedef typename SpatialObjectType::ConstPointer SpatialObjectConstPointer;

  /** Standard matrix type within this class. */
  typedef Matrix< ScalarType,
                  itkGetStaticConstMacro(ImageDimension),
                  itkGetStaticConstMacro(ImageDimension) >   MatrixType;

  /** Standard image type within this class. */
  typedef TImage ImageType;
  typedef double  MomentValueType;

  /** Standard image type pointer within this class. */
  typedef typename ImageType::Pointer      ImagePointer;
  typedef typename ImageType::ConstPointer ImageConstPointer;

  /** Affine transform for mapping to and from principal axis */
//  typedef AffineTransform< double, itkGetStaticConstMacro(ImageDimension) > AffineTransformType;
//  typedef typename AffineTransformType::Pointer                             AffineTransformPointer;

  /** Set the input image. */
  virtual void SetImage(const ImageType *image)
  {
    if ( m_Image != image )
      {
      m_Image = image;
      this->Modified();
      m_Valid = false;
      }
  }

  /** Set the spatial object mask. */
  virtual void SetSpatialObjectMask(const SpatialObject< itkGetStaticConstMacro(ImageDimension) > *so)
  {
    if ( m_SpatialObjectMask != so )
      {
      m_SpatialObjectMask = so;
      this->Modified();
      m_Valid = false;
      }
  }

  /** Compute moments of a new or modified image.
   * This method computes the moments of the image given as a
   * parameter and stores them in the object.  The values of these
   * moments and related parameters can then be retrieved by using
   * other methods of this object. */
  void Compute(void);

  itkGetMacro(I1,MomentValueType);
  itkGetMacro(I2,MomentValueType);
  itkGetMacro(I3,MomentValueType);
  itkGetMacro(I4,MomentValueType);
  itkGetMacro(I5,MomentValueType);
  itkGetMacro(I6,MomentValueType);
  itkGetMacro(I7,MomentValueType);
  itkGetMacro(Mass,MomentValueType);

protected:

  HuMomentsCalculator();
  ~HuMomentsCalculator() { }
  void PrintSelf(std::ostream & os, Indent indent) const;

private:
  HuMomentsCalculator(const Self &); //purposely not implemented
  void operator=(const Self &);         //purposely not implemented

  ImageConstPointer         m_Image;
  SpatialObjectConstPointer m_SpatialObjectMask;

  MomentValueType           m_I1;
  MomentValueType           m_I2;
  MomentValueType           m_I3;
  MomentValueType           m_I4;
  MomentValueType           m_I5;
  MomentValueType           m_I6;
  MomentValueType           m_I7;
  MomentValueType           m_Mass;
  bool                      m_Valid;



};

}
#ifndef ITK_MANUAL_INSTANTIATION
#include "itkHuMomentsCalculator.txx"
#endif

#endif // ITKHUMOMENTSCALCULATOR_H
