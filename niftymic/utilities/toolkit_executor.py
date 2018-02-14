##
# \file toolkit_executor.py
# \brief      generates function calls to execute other reconstruction toolkits
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Jan 2018
#

import os
import pysitk.python_helper as ph

EXE_IRTK = {
    "workstation": "/home/mebner/Development/VolumetricReconstruction_ImperialCollege/source/bin/SVRreconstructionGPU"
}


##
# Class to generate functions calls to execute other reconstruction toolkits
# \date       2018-01-27 02:12:00+0000
#
class ToolkitExecuter(object):

    def __init__(self, paths_to_images, paths_to_masks, dir_output):
        self._paths_to_images = paths_to_images
        self._paths_to_masks = paths_to_masks
        self._dir_output = dir_output

        # separator for command line export
        self._sep = " \\\n"

        self._subdir_temp = "./temp"

    ##
    # Gets the function call for fetalReconstruction toolkit provided by
    # Bernhard Kainz.
    # \date       2018-01-27 02:12:26+0000
    #
    # \param      self         The object
    # \param      option_args  The option arguments
    # \param      exe          The executable
    # \param      output_name  The output name
    #
    # \return     The function call irtk.
    #
    def get_function_call_irtk(
        self,
        option_args=['-d 0', '--useCPU', '--resolution 1'],
        exe=None,
        output_name="IRTK_SRR.nii.gz",
        kernel_mask_dilation=None,
    ):
        if exe is None:
            exe = EXE_IRTK["workstation"]

        cmd_args = []

        # store pwd
        cmd_args.append("PWD=$(pwd)")

        # change to output directory
        cmd_args.append("echo 'Change to output directory'")
        cmd_args.append("mkdir -p %s" % self._dir_output)
        cmd_args.append("cd %s" % self._dir_output)

        # create temp directory if required
        cmd_args.append("echo 'Create temp directory'")
        cmd_args.append("mkdir -p %s" % self._subdir_temp)

        # dilate masks
        if kernel_mask_dilation is not None:
            cmd_args.append("echo 'Dilate masks'")
            cmd_args.append(self._exe_dilate_masks(
                kernel_mask_dilation, self._paths_to_masks))

        # exe to determine slice thickness for toolkit
        cmd_args.append("echo 'Fetch slice thickness for all stacks'")
        cmd_args.append(self._exe_to_fetch_slice_thickness(self._paths_to_images))
        
        # toolkit execution
        cmd_args.append("echo 'IRTK Toolkit Execution'")
        exe_args = [exe]
        exe_args.append("-o %s" % output_name)
        exe_args.append("-i %s%s" %
                        (self._sep, self._sep.join(self._paths_to_images)))
        # exe_args.append("--manualMask %s" % self._paths_to_masks[0])  #
        # causes cuda sync error!?
        exe_args.append("-m %s%s" %
                        (self._sep, self._sep.join(self._paths_to_masks)))
        exe_args.append("--thickness `printf \"%s\" \"${thickness}\"`")
        exe_args.extend(option_args)
        toolkit_execution = "%s" % self._sep.join(exe_args)
        cmd_args.append(toolkit_execution)
        
        cmd_args.append("echo 'Delete temp directory'")
        cmd_args.append("rm -rf %s" %self._subdir_temp)

        cmd_args.append("echo 'Change back to original directory'")
        cmd_args.append("cd ${PWD}")
        cmd_args.append("\n")

        cmd = (" \n").join(cmd_args)
        return cmd

    ##
    # Provide bash-commands to read out slice thickness on-the-fly
    #
    # Rationale: IRTK recon toolkit assumes otherwise a thickness of twice the
    # voxel spacing by default
    # \date       2018-01-27 02:12:52+0000
    #
    # \param      paths_to_images  The paths to images
    #
    # \return     bash command as string
    #
    def _exe_to_fetch_slice_thickness(self, paths_to_images):
        cmd_args = []
        cmd_args.append("args=()")
        cmd_args.append("for i in %s" % (self._sep.join(paths_to_images)))
        cmd_args.append("do")
        cmd_args.append(
            "t=$(fslhd ${i} | grep pixdim3 | awk -F ' ' '{print $2}')")
        cmd_args.append("args+=(\" ${t}\")")
        cmd_args.append("done")
        cmd_args.append("thickness=${args[@]}")
        cmd_args.append("")
        cmd = ("\n").join(cmd_args)
        return cmd

    ##
    # Provide bash-commands to read out slice thickness on-the-fly
    #
    # Rationale: IRTK recon toolkit assumes otherwise a thickness of twice the
    # voxel spacing by default
    # \date       2018-01-27 02:12:52+0000
    #
    # \param      paths_to_images  The paths to images
    #
    # \return     bash command as string
    # #
    def _exe_dilate_masks(self, kernel, paths_to_masks, label=1):
        cmd_loop = []
        kernel_str = [str(k) for k in kernel]

        # Export dilated mask to temp directory
        for i_mask, path_to_mask in enumerate(paths_to_masks):
            directory = os.path.dirname(path_to_mask)
            mask_filename = os.path.basename(path_to_mask)
            path_to_mask_dilated = os.path.join(
                self._subdir_temp, ph.append_to_filename(mask_filename, "_dil"))

            cmd_args = ["c3d"]
            cmd_args.append(path_to_mask)
            cmd_args.append("-dilate %s %smm" % (label, "x".join(kernel_str)))
            cmd_args.append("-o %s" % path_to_mask_dilated)
            cmd = self._sep.join(cmd_args)

            cmd_loop.append(cmd)
            paths_to_masks[i_mask] = path_to_mask_dilated
        return "\n".join(cmd_loop)
