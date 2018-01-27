##
# \file toolkit_executor.py
# \brief      generates function calls to execute other reconstruction toolkits
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Jan 2018
#

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
        option_args=['--useCPU', '--resolution 1.25', '--thickness 4'],
        exe=None,
        output_name="IRTK_SRR.nii.gz",
    ):
        if exe is None:
            exe = EXE_IRTK["workstation"]

        exe_args = []
        exe_args.append("-o %s" % output_name)
        exe_args.append("-i %s" % " ".join(self._paths_to_images))
        # exe_args.append("--manualMask %s" % self._paths_to_masks[0])  # causes cuda sync error!?
        exe_args.append("-m %s" % " ".join(self._paths_to_masks))
        exe_args.append("--thickness `printf \"%s\" \"${thickness}\"`")
        exe_args.append(" ".join(option_args))
        toolkit_execution = "%s %s" % (exe, " ".join(exe_args))

        cmd_args = []
        cmd_args.append("PWD=$(pwd)")
        cmd_args.append("mkdir -p %s" % self._dir_output)
        cmd_args.append("cd %s" % self._dir_output)
        cmd_args.append("%s" %
                        self._exe_to_fetch_slice_thickness(self._paths_to_images))
        cmd_args.append(toolkit_execution)
        cmd_args.append("")
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
    @staticmethod
    def _exe_to_fetch_slice_thickness(paths_to_images):
        cmd_args = []
        cmd_args.append("args=()")
        cmd_args.append("for i in %s" % (" ".join(paths_to_images)))
        cmd_args.append("do")
        cmd_args.append(
            "t=$(fslhd ${i} | grep pixdim3 | awk -F ' ' '{print $2}')")
        cmd_args.append("args+=(\" ${t}\")")
        cmd_args.append("done")
        cmd_args.append("thickness=${args[@]}")
        cmd_args.append("")
        cmd = ("\n").join(cmd_args)
        return cmd

    
