import numpy as np
from .sigproc import magspec, frame_signal
import wavio
import math
from math import ceil


def getFrames(audioFile, frame_time_span = 8, step_time_span = 2,
                   start_time = 0, end_time=-1, window_fn = None, get_sequence = False):
    '''Frames a portion of the pressure measurements from an audio file.'''
    
    freq_resolution = 1000 / frame_time_span

    # Load audio file
    if type(audioFile) == wavio.Wav:
        wav_data = audioFile
    else:
        wav_data = wavio.read(audioFile)
    
    # Rescale data if sample width is > 2
    if wav_data.sampwidth > 2:
            wav_data.data //= 2 ** (8 * (wav_data.sampwidth - 2))

    # #
    # Split the wave signal into overlapping frames
    # #

    start_frame = int(start_time / 1000 * wav_data.rate)
    end_frame = int((end_time / 1000 + frame_time_span / 1000 - step_time_span / 1000)* wav_data.rate)


    frame_sample_span = int(math.floor(frame_time_span / 1000 * wav_data.rate))
    step_sample_span = step_time_span / 1000 * wav_data.rate
    # No frames if the audio file is too short
    if wav_data.data[start_frame:end_frame].shape[0] < frame_sample_span:
        frames = []
        return np.array(frames)
    else:
        frames = frame_signal(wav_data.data.ravel()[start_frame:end_frame], frame_sample_span, step_sample_span)
    
    if window_fn != None:
       frames = frames * window_fn(frames.shape[1]) 
    
    if get_sequence:
        return frames, wav_data.data.ravel()[start_frame:end_frame]
    return frames

def getComplexSpectrogram(audioFile, frame_time_span = 8, step_time_span = 2,
        min_freq = 5000, max_freq = 50000,
        start_time = 0, end_time=-1, window_fn = None):
    '''
    Gets and returns a two-dimensional list in which the values encode a spectrogram.

    :param audioFile: the audio file in .wav format for which a spectrogram is generated.
        This may either be an audio file of type wavio.Wav or a file name
    :param frame_time_span: ms, length of time for one time window for dft
    :param step_time_span: ms, length of time step for spectrogram
    :param spec_clip_min: log magnitude spectrogram min-max normalization, minimum value
    :param spec_clip_max: log magnitude spectrogram min-max normalization, maximum value
    :param min_freq: Hz, lower bound of frequency for spectrogram
    :param max_freq: Hz, upper bound of frequency for spectrogram
    :param start_time: ms, the beginning of where the audioFile is read
    :param end_time: ms, the end of where the audioFile is read. If end > the length of
        of the file, then the file is read only to its end.
    :param window_fn: the function that generates a processing map for each frame
        before the the frames are used in the spectrogram. The function must receive
        one positional argument, n, and then return an array of length n.
        For example, window_fn(5) could return [0.1,0.2,0.4,0.2,0.1]
    :returns: A tuple with both the spectrogram and the time at which the
        spectrogram ended in ms: (spectogram, end_time)
    '''

    freq_resolution = 1000 / frame_time_span

    frames = getFrames(audioFile,
            frame_time_span = frame_time_span,
            step_time_span = step_time_span,
            start_time = start_time,
            end_time = end_time, window_fn = window_fn)

    # No spectrogram if the audio file is too short
    if len(frames) == 0:
        return np.array([[]], dtype=float), start_time
    
    # #
    # Make spectrogram
    # #
    NFFT = len(frames[0])

    # Compute magnitude spectra
    singal_magspec = np.fft.rfft(frames, NFFT)

    # Include only the desired frequency range
    clip_bottom = int(min_freq // freq_resolution)
    clip_top = int(max_freq // freq_resolution) 
    spectrogram = singal_magspec.T[clip_bottom:clip_top]

    # Flip spectrogram to match expectations for display
    # Also normalize
    # spectrogram = spectrogram[::-1,]

    actual_end_time = start_time + spectrogram.shape[1] * step_time_span
    return spectrogram, actual_end_time


def getSpectrogram(audioFile, frame_time_span = 8, step_time_span = 2, spec_clip_min = 0,
                   spec_clip_max = 6, min_freq = 5000, max_freq = 50000,
                   start_time = 0, end_time=-1, window_fn = None, return_db = False):
    '''
    Gets and returns a two-dimensional list in which the values encode a spectrogram.

    :param audioFile: the audio file in .wav format for which a spectrogram is generated.
        This may either be an audio file of type wavio.Wav or a file name
    :param frame_time_span: ms, length of time for one time window for dft
    :param step_time_span: ms, length of time step for spectrogram
    :param spec_clip_min: log magnitude spectrogram min-max normalization, minimum value
    :param spec_clip_max: log magnitude spectrogram min-max normalization, maximum value
    :param min_freq: Hz, lower bound of frequency for spectrogram
    :param max_freq: Hz, upper bound of frequency for spectrogram
    :param start_time: ms, the beginning of where the audioFile is read
    :param end_time: ms, the end of where the audioFile is read. If end > the length of
        of the file, then the file is read only to its end.
    :param window_fn: the function that generates a processing map for each frame
        before the the frames are used in the spectrogram. The function must receive
        one positional argument, n, and then return an array of length n.
        For example, window_fn(5) could return [0.1,0.2,0.4,0.2,0.1]
    :param return_db: if True, the returned spectrogram is in DB scale; else, the
        returned spectrogram is normalized
    :returns: A tuple with both the spectrogram and the time at which the
        spectrogram ended in ms: (spectogram, end_time)
    '''

    freq_resolution = 1000 / frame_time_span

    frames = getFrames(audioFile,
            frame_time_span = frame_time_span,
            step_time_span = step_time_span,
            start_time = start_time,
            end_time = end_time, window_fn = window_fn)
    
    # If there was not a long enough segment to form a whole frame,
    # raise an error
    if frames.size == 0:
        raise ValueError(f"""
        A long enough segment of audio samples was not available to calculate even a single discrete fourier transform 
        {start_time=}, {end_time=}
        """)

    # #
    # Make spectrogram
    # #
    NFFT = len(frames[0])

    # Compute magnitude spectra
    singal_magspec = magspec(frames, NFFT)

    # Include only the desired frequency range
    clip_bottom = int(min_freq // freq_resolution)
    clip_top = int(max_freq // freq_resolution) 
    spectrogram = singal_magspec.T[clip_bottom:clip_top]
    spectrogram = np.log10(spectrogram)

    # Flip spectrogram to match expectations for display
    spectrogram = spectrogram[::-1,]

    actual_end_time = start_time + spectrogram.shape[1] * step_time_span

    if return_db:
        return 20 * spectrogram, actual_end_time

    # normalize 0-1
    spectrogram = normalize3(spectrogram, spec_clip_min, spec_clip_max)

    return spectrogram, actual_end_time

def getAnnotationMask(annotations, frame_time_span = 8, step_time_span = 2,
                      min_freq = 5000, max_freq = 50000, start_time = 0,
                      end_time=-1, line_thickness = 1):
    '''
    Gets and returns a two-dimensional list in which the values encode a mask of the annotations.
    The generated mask will have the same shape as will a spectrogram generated by getSpectrogram
    with equivalent parameters.

    :param annotations: The two dimensional array with contours on the first axis and with
                        (time_s,freq_hz) nodes on the second axis. As returned from
                        tonalReader.getTimeFrequencyContours().
    :param frame_time_span: ms, length of time for one time window for dft
    :param step_time_span: ms, length of time step for spectrogram
    :param min_freq: Hz, lower bound of frequency for spectrogram
    :param max_freq: Hz, upper bound of frequency for spectrogram
    :param start_time: ms, the beginning of where the audioFile is read
    :param end_time: ms, the end of where the audioFile is read. -1 reads until the end
    :param line_thickness: the number of pixels, i.e. frequency
                           bins, tall that the annotations will be

    :returns: annotation mask
    '''

    freq_resolution = 1000 / frame_time_span
    
    # The spectrogram must start at min_freq = 0. We therefore need to readjust the min and max frequencies
    max_freq = max_freq - (max_freq % freq_resolution)
    min_freq = min_freq - (min_freq % freq_resolution)

    # Get dimensions for mask
    image_width = int((end_time - start_time) / step_time_span)
    image_height = int((max_freq - min_freq) / freq_resolution)

    mask = np.zeros((image_height, image_width))

    # Get only the annotations that will be present in the mask
    annotations = [a for a in annotations if a[-1][0] >= start_time/ 1000 and a[0][0] < end_time/1000]


    # if no annotations to plot
    if len(annotations) == 0:
         return mask

    time_span = (end_time - start_time)

    # plot the portions of annotations that are within the time-frequency range
    for annotation in annotations:
        prev_time_frame = 0
        prev_freq_frame = 0
        first_flag = True
        for time, freq in annotation:
            # get approximate pixel frame for timestamp & frequency
            time_frame = (time*1000 - start_time) * image_width / time_span
            freq_frame = (max_freq - freq) / freq_resolution

            if first_flag:
                prev_time_frame = time_frame
                prev_freq_frame = freq_frame
                first_flag = False
                continue
            
            # If the time frame is above image width,
            # all future ones will be in this annotation
            if prev_time_frame >= image_width:
                break
            
            # If time frame is before the image
            if time_frame < -0.5:
                continue

            # If both are prev and curr are outside image
            if ((freq_frame < -0.5 and prev_freq_frame < -0.5) or
                    (freq_frame >= image_height and prev_freq_frame >= image_height)):
                continue

            # Interpolating line function
            freq_time_line = (
                lambda x: freq_frame + (prev_freq_frame - freq_frame) / 
                (prev_time_frame - time_frame)*(x - time_frame) )

            distance = np.sqrt((time_frame-prev_time_frame)**2 + (freq_frame - prev_freq_frame)**2)
            # Draw interpolating line
            for t in np.linspace(prev_time_frame, time_frame, math.ceil(distance) + 1):
                
                t_rounded = round(t)
                # check that time is within the image
                if t_rounded < 0 or t_rounded >= image_width:
                    continue

                # get frequency from interpolation line.
                if time_frame - prev_time_frame < 1e-10:
                    curr_freq_rounded = round(freq)
                else:
                    curr_freq_rounded = round(freq_time_line(t))
                
                # Check that frequency is within the image
                if curr_freq_rounded < 0 or curr_freq_rounded >= image_height:
                    continue
                
                # Draw pixel
                mask[max(curr_freq_rounded - line_thickness//2,0): curr_freq_rounded + ceil(line_thickness/2), t_rounded] = 1

            prev_time_frame = time_frame
            prev_freq_frame = freq_frame
    
    return mask

def expand_annotation_mask(annotation_mask, spectrogram,
        threshold = 0.9, max_distance = 5, min_snr = 0.98):
    '''Widens the an annotation mask based on the energy levels in
    the corresponding spectrogram.

    :param annotation_mask: an annotation mask as generated by getAnnotationMask
    :param spectrogram: a spectrogram as generated by getSpectrogram
    :param threshold: a floating point value. The ratio between the energy of a candidate pixel and
        the known tonal energy from which the candidate is extending  must be at least the value of threshold
    :param max_distance: the maximum distance, in pixels, from the
        annotation allowable for expansion. Setting this to 1 results in no
        expansion, to 2 results in at maximum only one pixel wider and so on
    :param min_snr: if not None, the ratio between the energy of a candidate pixel and that of the
        background noise must be above min_snr
    '''

    mask = annotation_mask.copy()
    
    # if no tonal energy
    if mask.sum()== 0:
        return mask
    
    background_energy = spectrogram[np.where(mask == 0)].sum()/(mask==0).sum()

    for i,j in np.ndindex(mask.shape):
        if annotation_mask[i,j] == 0:
            continue
        min_j = max(0, j - max_distance)
        max_j = min(j+max_distance, mask.shape[1])
        
        # Attempt to widen annotation pixel in the mask both to the left
        # and to the right
        left_and_right = [range(j-1, min_j - 1, -1), range(j+1, max_j)]
        
        tonal_energy = spectrogram[i,j]
        if tonal_energy == 0:
            continue
        for direction in left_and_right:
            for jp in direction:
                # if the mask already has energy here,
                # continue to the next 
                if annotation_mask[i,jp] == 1:
                    continue
                # if there is a drop off of energy from the tonal energy 
                if min_snr != None and spectrogram[i,jp]/background_energy < min_snr:
                    break
                if spectrogram[i,jp]/tonal_energy < threshold:
                    break
                
                mask[i,jp] = 1

    return mask


####### UTILITY #######

# Credit to Pu Li https://github.com/Paul-LiPu/DeepWhistle
# min-max normalization
def normalize3(mat, min_v, max_v):
    mat = np.clip(mat, min_v, max_v)
    return (mat - min_v) / (max_v - min_v)

