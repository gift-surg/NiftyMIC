/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkScaledTranslationEuler3DTransform_h
#define itkScaledTranslationEuler3DTransform_h

#include <iostream>
#include "itkEuler3DTransform.h"
#include "itkMacro.h"
#include "itkVersor.h"

namespace itk
{
/** \class ScaledTranslationEuler3DTransform
 * \brief ScaledTranslationEuler3DTransform of a vector space (e.g. space coordinates)
 *
 * This transform applies a rotation about a specific coordinate or
 * centre of rotation followed by a translation.
 *
 * \ingroup ITKTransform
 */
template<typename TParametersValueType=double>
class ScaledTranslationEuler3DTransform :
  public Euler3DTransform<TParametersValueType>
{
public:
  /** Standard class typedefs. */
  typedef ScaledTranslationEuler3DTransform      Self;
  typedef Euler3DTransform<TParametersValueType> Superclass;
  typedef SmartPointer<Self>                     Pointer;
  typedef SmartPointer<const Self>               ConstPointer;

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ScaledTranslationEuler3DTransform, Euler3DTransform);

  /** Dimension of the space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, 3);
  itkStaticConstMacro(InputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(ParametersDimension, unsigned int, 9);

  typedef typename Superclass::ParametersType            ParametersType;
  typedef typename Superclass::ParametersValueType       ParametersValueType;
  typedef typename Superclass::FixedParametersType       FixedParametersType;
  typedef typename Superclass::FixedParametersValueType  FixedParametersValueType;
  typedef typename Superclass::JacobianType              JacobianType;
  typedef typename Superclass::ScalarType                ScalarType;
  typedef typename Superclass::InputVectorType           InputVectorType;
  typedef typename Superclass::OutputVectorType          OutputVectorType;
  typedef typename Superclass::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;

  typedef typename Superclass::InputVnlVectorType   InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType  OutputVnlVectorType;
  typedef typename Superclass::InputPointType       InputPointType;
  typedef typename Superclass::OutputPointType      OutputPointType;
  typedef typename Superclass::MatrixType           MatrixType;
  typedef typename Superclass::InverseMatrixType    InverseMatrixType;
  typedef typename Superclass::CenterType           CenterType;
  typedef typename Superclass::TranslationType      TranslationType;
  typedef typename Superclass::TranslationValueType TranslationValueType;
  typedef typename Superclass::OffsetType           OffsetType;
  
  /** Base inverse transform type. This type should not be changed to the
   * concrete inverse transform type or inheritance would be lost. */
  typedef typename Superclass::InverseTransformBaseType InverseTransformBaseType;
  typedef typename InverseTransformBaseType::Pointer    InverseTransformBasePointer;

  /** Set the transformation from a container of parameters
   * This is typically used by optimizers.  There are nine parameters. The first
   * three represent the angles of rotation (in radians) around each one of the
   * axes (X,Y,Z), the next three parameters represent the coordinates of the
   * center of rotation and the last three parameters represent the
   * translation. */
  void SetParameters(const ParametersType & parameters) ITK_OVERRIDE;

  /** Get the parameters that uniquely define the transform
   * This is typically used by optimizers. There are nine parameters. The first
   * three represent the angles of rotation (in radians) around each one of the
   * axes (X,Y,Z), the next three parameters represent the coordinates of the
   * center of rotation and the last three parameters represent the
   * translation. */
  // const ParametersType & GetParameters(void) const ITK_OVERRIDE;

  // const OutputVectorType & GetTranslation(void) const ITK_OVERRIDE;


  /** This method computes the Jacobian matrix of the transformation.
   * given point or vector, returning the transformed point or
   * vector. The rank of the Jacobian will also indicate if the
   * transform is invertible at this point. */
  virtual void ComputeJacobianWithRespectToParameters( const InputPointType  & p, JacobianType & jacobian) const ITK_OVERRIDE;

  /** Get an inverse of this transform. */
  // bool GetInverse(Self *inverse) const;

  /** Return an inverse of this transform. */
  // virtual InverseTransformBasePointer GetInverseTransform() const ITK_OVERRIDE;

  /**
   * Set/Get TranslationScale
   */
  virtual void SetTranslationScale( const ScalarType TranslationScale )
    {
    itkDebugMacro( "setting TranslationScale to " << TranslationScale );
    if( Math::NotExactlyEquals(this->m_TranslationScale, TranslationScale) )
      {
      this->m_TranslationScale = TranslationScale;
      // std::cout << "TranslationScale is set to " << this->m_TranslationScale << std::endl;
      this->Modified();
      }
    }
  itkGetConstMacro( TranslationScale, ScalarType );


protected:
  ScaledTranslationEuler3DTransform();
  ScaledTranslationEuler3DTransform(const MatrixType & matrix, const OutputPointType & offset);
  ScaledTranslationEuler3DTransform(unsigned int ParametersDimension);
  ~ScaledTranslationEuler3DTransform();

  /**
   * Print contents of an ScaledTranslationEuler3DTransform
   */
  // void PrintSelf(std::ostream & os, Indent indent) const ITK_OVERRIDE;

private:
  ScaledTranslationEuler3DTransform(const Self &) ITK_DELETE_FUNCTION;
  void operator=(const Self &) ITK_DELETE_FUNCTION;
  ScalarType m_AngleX;
  ScalarType m_AngleY;
  ScalarType m_AngleZ;
  bool       m_ComputeZYX;
  ScalarType                                  m_TranslationScale;


};                                        // class ScaledTranslationEuler3DTransform
}  // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkScaledTranslationEuler3DTransform.hxx"
#endif

#endif /* itkScaledTranslationEuler3DTransform_h */
