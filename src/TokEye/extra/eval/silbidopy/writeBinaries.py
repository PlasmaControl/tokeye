"""
Write annotations
Source: https://github.com/joshua-zingale/silbidopy/blob/master/silbidopy/writeBinaries.py
"""

import struct
import dataclasses

SHORT_LEN = 2
INT_LEN = 4
DOUBLE_LEN = 8
LONG_LEN = 8


HEADER_STR = "silbido!".encode("utf-8")
DET_VERSION = 4

# feature bit-mask - describes what has been populated and allows
# backward compatibility
    
# features produced for every point
TIME = 1
FREQ = 1 << 1
SNR = 1 << 2
PHASE = 1 << 3
    
RIDGE = 1 << 6

TIMESTAMP = 1 << 7  # base timestamp for detections
USERCOMMENT = 1 << 8  # user comment field
    
# features produced once per call
SCORE = 1 << 4
CONFIDENCE = 1 << 5
SPECIES = 1 << 9
CALL = 1 << 10

DEFAULT = TIME | FREQ


def writeContoursBinary(filename, contours,
        time = True, frequency = True, snr = False,
        phase = False, ridge = False,
        comment = "", timestamp = "", score = False,
        confidence = False, species = False, call = False):
    '''Writes contours to a silbido binary file.

    UserVersion is currently set to 0.
    The fields that are to be present in this annotation file must be specified
    in the keyword arguments, which all, save comment and timestamp, are boolean.
    comment and timestamp being the empty string means that there will not be saved
    a comment or timestamp respectively. The graphid is set to an arbitrary constant value
    for each tonal.
    
    :param filename: the name of the file to which to be written
    :param contours: an array containing dictionaries, each of which represents on contour
        Each dictionary must contain the fields specified in the keyword arguments.
        If all fields are specified, then the dictionary would look as follows:
        {"score": 0.8, "confidence": 1.0, "species": "Baleen Whale", "call": "D",
        "tfnodes": [
            {"time": 3.25, "freq": 50.125, "snr": 6.6, "phase": 0.25, "ridge": 1.0},
            {"time":...},
            ...,
        ]}
    '''
    
    version = DET_VERSION.to_bytes(SHORT_LEN, byteorder = "big")
    bitMask = 0
    
    headerSize = 3 * SHORT_LEN + INT_LEN + len(HEADER_STR)
    
    if time:
        bitMask += TIME
    if frequency:
        bitMask += FREQ
    if snr:
        bitMask += SNR
    if phase:
        bitMask += PHASE
    if ridge:
        bitMask += RIDGE
    if len(comment) > 0:
        bitMask += USERCOMMENT
        headerSize += 2 + len(comment)
    if len(timestamp) > 0:
        bitMask += TIMESTAMP
        headerSize += 2 + len(timestamp)
    if score:
        bitMask += SCORE
    if confidence:
        bitMask += CONFIDENCE
    if species:
        bitMask += SPECIES
    if call:
        bitMask += CALL

    bitMask = bitMask.to_bytes(SHORT_LEN, byteorder="big")
    userVersion = (0).to_bytes(SHORT_LEN, byteorder="big")
    headerSize = headerSize.to_bytes(INT_LEN, byteorder="big")

    file = open(filename, 'wb')

    # Write magic string
    file.write(HEADER_STR)

    # Write header
    file.write(version)
    file.write(bitMask)
    file.write(userVersion)
    file.write(headerSize)
    if comment:
        write_utf8_string(file, comment)
    if timestamp:
        write_utf8_string(timestamp)

    # Write tonal meta deta
    # graphId is an arbitrary value as of now
    graphId = (14567891234567891234).to_bytes(LONG_LEN, byteorder = "big")

    # Write tonals  
    for contour in contours:
        
        if confidence:
            file.write(struct.pack(">d", contour["confidence"]))
        if score:
            file.write(struct.pack(">d", contour["score"]))
        if species:
            L = len(contour["species"])
            if L >= 2**16:
                raise RuntimeError("Length of species name is too long")

            file.write(L.to_bytes(SHORT_LEN, byteorder="big"))
            file.write(contour["species"].encode("utf-8"))
        if call:
            L = len(contour["call"])
            if L >= 2**16:
                raise RuntimeError("Length of call name is too long")

            file.write(L.to_bytes(SHORT_LEN, byteorder="big"))
            file.write(contour["call"].encode("utf-8"))
        
        file.write(graphId)

        N = len(contour["tfnodes"]).to_bytes(INT_LEN, byteorder = "big") # number of points in contour
        
        file.write(N)

        # Write all time and frequency nodes for the current contour
        for node in contour["tfnodes"]:
            if time:
                file.write(struct.pack('>d', node["time"]))
            if frequency:
                file.write(struct.pack('>d', node["freq"]))
            if snr:
                file.write(struct.pack(">d", node["snr"]))
            if phase:
                file.write(struct.pack(">d", node["phase"]))
            if ridge:
                file.write(struct.pack(">d", node["ridge"]))



def writeTimeFrequencyBinary(filename, contours, userVersion=0):
    '''Writes only time and frequency and leaves no comment nor timestamp.

    :param filename: the name of the file to which to be written
    :param contours: Each contour is either
                     1.  a dataclass with mandatory fields:
                         time - array of offsets in s
                         freq - array of frequencies in Hz
                         and optional fields (must be in every contour if
                         present):
                         species - species name
                         call - call name
                      or
                     2.  A list of (time, frequency) tuples
                         e.g. [[(1.2,75.23), (1.25, 74.77)],
                               [(4.9,62.48), (5.52, 60.29)]
                              ]
    :param userVersion:  User's internal version number (e.g. for multiple
        annotations of the same file)
    '''

    # Determine what will be written

    # See if we should be expecting dataclass fields
    expect_fields = False
    fields = []
    if len(contours) > 0:
        # Assume contours is homogeneous, infer structure from first contour
        expect_fields = dataclasses.is_dataclass(contours[0])
        # Retrieve field names
        fields = [f.name for f in dataclasses.fields(contours[0])]
        if "time" not in fields or "freq" not in fields:
            raise RuntimeError("Contours must have time and freq fields")

    bitMask = (TIME | FREQ)
    if "species" in fields:
        bitMask |= SPECIES
    if "call" in fields:
        bitMask |= CALL

    # ASSUMES no comments
    headerSize = (3 * SHORT_LEN + INT_LEN +
                  len(HEADER_STR)).to_bytes(INT_LEN, byteorder = "big")

    file = open(filename, 'wb')

    # Write magic string
    file.write(HEADER_STR)

    # Write header
    file.write(DET_VERSION.to_bytes(SHORT_LEN, byteorder="big"))
    file.write(bitMask.to_bytes(SHORT_LEN, byteorder="big"))
    file.write(userVersion.to_bytes(SHORT_LEN, byteorder="big"))
    file.write(headerSize)

    # Write tonal meta deta

    # Specific to graph-based processing described in "Automated
    # extraction of odontocete whistle contours, Roch et al. 2011
    # DOI: 10.1121/1.3624821.  Unused here.
    graphId = (0).to_bytes(LONG_LEN, byteorder="big")

    # Write tonals  
    for contour in contours:

        # number of points in contour
        if expect_fields:
            N = len(contour.time)
        else:
            N = len(contour)

        if expect_fields:
            # contour is a dataclass with fields
            if bitMask & SPECIES:
                write_utf8_string(file, contour.species)
            if bitMask & CALL:
                write_utf8_string(file, contour.call)

        file.write(graphId)
        file.write(N.to_bytes(INT_LEN, byteorder="big"))

        # Write all time and frequency nodes for the current contour
        if expect_fields:
            # contour is a dataclass with fields
            if bitMask & TIME and bitMask & FREQ:
                for idx in range(N):
                    file.write(struct.pack('>d', contour.time[idx]))
                    file.write(struct.pack('>d', contour.freq[idx]))
        else:
            # List of iterables (e.g., tuples)
            for time, freq in contour:
                file.write(struct.pack('>d', time))
                file.write(struct.pack('>d', freq))


def write_utf8_string(file, strval):
    """

    :param file:  handle to output file
    :param strval:  text to write
    :return: None
    """
    n = len(strval)
    if n > 2 ** 16:
        raise RuntimeError("Length of string is too long")
    file.write(n.to_bytes(SHORT_LEN, byteorder="big"))
    file.write(strval.encode("utf-8"))