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
#ifndef itkScaledTranslationEuler3DTransform_hxx
#define itkScaledTranslationEuler3DTransform_hxx

#include "itkScaledTranslationEuler3DTransform.h"

namespace itk
{
// Constructor with default arguments
template<typename TParametersValueType>
ScaledTranslationEuler3DTransform<TParametersValueType>::ScaledTranslationEuler3DTransform() :
  Superclass(ParametersDimension)
{
  m_ComputeZYX = false;
  m_AngleX = m_AngleY = m_AngleZ = NumericTraits<ScalarType>::ZeroValue();
  this->m_TranslationScale = 1.0;
}

// Constructor with default arguments
template<typename TParametersValueType>
ScaledTranslationEuler3DTransform<TParametersValueType>::ScaledTranslationEuler3DTransform(unsigned int parametersDimension) :
  Superclass(parametersDimension)
{
  m_ComputeZYX = false;
  m_AngleX = m_AngleY = m_AngleZ = NumericTraits<ScalarType>::ZeroValue();
  this->m_TranslationScale = 1.0;
}

// Constructor with default arguments
template<typename TParametersValueType>
ScaledTranslationEuler3DTransform<TParametersValueType>::ScaledTranslationEuler3DTransform(const MatrixType & matrix,
                                                                const OutputPointType & offset) :
  Superclass(matrix, offset)
{
  m_ComputeZYX = false;
  this->SetMatrix(matrix);

  OffsetType off;
  off[0] = offset[0];
  off[1] = offset[1];
  off[2] = offset[2];
  this->SetOffset(off);

  this->m_TranslationScale = 1.0;
}

// Destructor
template<typename TParametersValueType>
ScaledTranslationEuler3DTransform<TParametersValueType>::
~ScaledTranslationEuler3DTransform()
{
}

//
// Set Parameters
//
// Parameters are ordered as:
//
// p[0:2] = rotations about x, y and z axes
// p[3:5} = center of rotation
// p[6:8] = translation
//
//
// template<typename TParametersValueType>
// void
// ScaledTranslationEuler3DTransform<TParametersValueType>
// ::SetParameters(const ParametersType & parameters)
// {

//   ParametersType scaledparams = parameters;
//   // this->m_TranslationScale = 1.0;

//   // Scale Translation
//   scaledparams[3] *= this->m_TranslationScale;
//   scaledparams[4] *= this->m_TranslationScale;
//   scaledparams[5] *= this->m_TranslationScale;

//   // // Set Parameters by using Superclass
//   this->Superclass::SetParameters(scaledparams);

//   // Set angles with parameters
//   // m_AngleX = scaledparams[0];
//   // m_AngleY = scaledparams[1];
//   // m_AngleZ = scaledparams[2];
//   // this->ComputeMatrix();

//   // // Transfer the translation part
//   // OutputVectorType newTranslation;
//   // newTranslation[0] = scaledparams[3];
//   // newTranslation[1] = scaledparams[4];
//   // newTranslation[2] = scaledparams[5];
//   // this->SetVarTranslation(newTranslation);
//   // this->ComputeOffset();

//   // // Modified is always called since we just have a pointer to the
//   // // parameters and cannot know if the parameters have changed.
//   // this->Modified();

//   // itkDebugMacro(<< "After setting parameters ");
// }

// //
// // Get Parameters
// //
// // Parameters are ordered as:
// //
// // p[0:2] = rotations about x, y and z axes
// // p[3:5} = center of rotation
// // p[6:8] = translation
// //

// template<typename TParametersValueType>
// const typename ScaledTranslationEuler3DTransform<TParametersValueType>::ParametersType
// & ScaledTranslationEuler3DTransform<TParametersValueType>
// ::GetParameters(void) const
//   {

//   this->m_Parameters[0] = m_AngleX;
//   this->m_Parameters[1] = m_AngleY;
//   this->m_Parameters[2] = m_AngleZ;
//   // this->m_Parameters[3] = this->GetTranslation()[0] / this->m_TranslationScale;
//   // this->m_Parameters[4] = this->GetTranslation()[1] / this->m_TranslationScale;
//   // this->m_Parameters[5] = this->GetTranslation()[2] / this->m_TranslationScale;
//   this->m_Parameters[3] = this->GetTranslation()[0];
//   this->m_Parameters[4] = this->GetTranslation()[1];
//   this->m_Parameters[5] = this->GetTranslation()[2];


//   return this->m_Parameters;
//   }

// template<typename TParametersValueType>
// void
// ScaledTranslationEuler3DTransform<TParametersValueType>
// ::ComputeJacobianWithRespectToParameters(const InputPointType & p, JacobianType & jacobian) const
// {
//   // need to check if angles are in the right order
//   const double cx = std::cos(m_AngleX);
//   const double sx = std::sin(m_AngleX);
//   const double cy = std::cos(m_AngleY);
//   const double sy = std::sin(m_AngleY);
//   const double cz = std::cos(m_AngleZ);
//   const double sz = std::sin(m_AngleZ);

//   jacobian.SetSize( 3, this->GetNumberOfLocalParameters() );
//   jacobian.Fill(0.0);

//   const double px = p[0] - this->GetCenter()[0];
//   const double py = p[1] - this->GetCenter()[1];
//   const double pz = p[2] - this->GetCenter()[2];

//   if( m_ComputeZYX )
//     {
//     jacobian[0][0] = ( cz * sy * cx + sz * sx ) * py + ( -cz * sy * sx + sz * cx ) * pz;
//     jacobian[1][0] = ( sz * sy * cx - cz * sx ) * py + ( -sz * sy * sx - cz * cx ) * pz;
//     jacobian[2][0] = ( cy * cx ) * py + ( -cy * sx ) * pz;

//     jacobian[0][1] = ( -cz * sy ) * px + ( cz * cy * sx ) * py + ( cz * cy * cx ) * pz;
//     jacobian[1][1] = ( -sz * sy ) * px + ( sz * cy * sx ) * py + ( sz * cy * cx ) * pz;
//     jacobian[2][1] = ( -cy ) * px + ( -sy * sx ) * py + ( -sy * cx ) * pz;

//     jacobian[0][2] = ( -sz * cy ) * px + ( -sz * sy * sx - cz * cx ) * py
//       + ( -sz * sy * cx + cz * sx ) * pz;
//     jacobian[1][2] = ( cz * cy ) * px + ( cz * sy * sx - sz * cx ) * py + ( cz * sy * cx + sz * sx ) * pz;
//     jacobian[2][2] = 0;
//     }
//   else
//     {
//     jacobian[0][0] = ( -sz * cx * sy ) * px + ( sz * sx ) * py + ( sz * cx * cy ) * pz;
//     jacobian[1][0] = ( cz * cx * sy ) * px + ( -cz * sx ) * py + ( -cz * cx * cy ) * pz;
//     jacobian[2][0] = ( sx * sy ) * px + ( cx ) * py + ( -sx * cy ) * pz;

//     jacobian[0][1] = ( -cz * sy - sz * sx * cy ) * px + ( cz * cy - sz * sx * sy ) * pz;
//     jacobian[1][1] = ( -sz * sy + cz * sx * cy ) * px + ( sz * cy + cz * sx * sy ) * pz;
//     jacobian[2][1] = ( -cx * cy ) * px + ( -cx * sy ) * pz;

//     jacobian[0][2] = ( -sz * cy - cz * sx * sy ) * px + ( -cz * cx ) * py
//       + ( -sz * sy + cz * sx * cy ) * pz;
//     jacobian[1][2] = ( cz * cy - sz * sx * sy ) * px + ( -sz * cx ) * py
//       + ( cz * sy + sz * sx * cy ) * pz;
//     jacobian[2][2] = 0;
//     }

//   // compute derivatives for the translation part
//   unsigned int blockOffset = 3;
//   for( unsigned int dim = 0; dim < SpaceDimension; dim++ )
//     {
//     jacobian[dim][blockOffset + dim] = 1.0 * this->m_TranslationScale;
//     }
// }

} // namespace

#endif
