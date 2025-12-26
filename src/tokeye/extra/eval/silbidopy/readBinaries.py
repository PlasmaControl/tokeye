"""
Read annotations
Source: https://github.com/joshua-zingale/silbidopy/blob/master/silbidopy/readBinaries.py
"""

import struct

SHORT_LEN = 2
INT_LEN = 4
DOUBLE_LEN = 8
LONG_LEN = 8

class TonalHeader:
    def __init__(self, file):
        '''Given a file object, type being "_io.BufferedReader", reads in
        the header of a silbido annotation file.
        '''

        self.HEADER_STR = "silbido!"

        self.DET_VERSION = 4

        # feature bit-mask - describes what has been populated and allows
        # backward compatibility

        # features produced for every point
        self.TIME = 1
        self.FREQ = 1 << 1
        self.SNR = 1 << 2
        self.PHASE = 1 << 3

        self.RIDGE = 1 << 6

        self.TIMESTAMP = 1 << 7  # base timestamp for detections
        self.USERCOMMENT = 1 << 8  # user comment field

        # features produced once per call
        self.SCORE = 1 << 4
        self.CONFIDENCE = 1 << 5
        self.SPECIES = 1 << 9
        self.CALL = 1 << 10

        self.DEFAULT = self.TIME | self.FREQ

        # length of header identifier string
        self.magicLen = len(self.HEADER_STR)

        # fields to be filled in
        self.comment = None
        self.timestamp = None
        self.timeFormatter = None

        self.userVersion = None
        self.bitMask = None
        self.headerSize = None
        self.version = None

        try:
            magicStr = str(file.read(self.magicLen), 'utf-8')
        except:
            magicStr = ''
        if magicStr == self.HEADER_STR:
            # found magic string, looks like a valid header
            self.version = int.from_bytes(file.read(SHORT_LEN), byteorder = "big")
            self.bitMask = int.from_bytes(file.read(SHORT_LEN), byteorder = "big")
            self.userVersion = int.from_bytes(file.read(SHORT_LEN), byteorder = "big")
            self.headerSize = int.from_bytes(file.read(INT_LEN), byteorder = "big")

            # Figure out how much of the header has already been read
            headerUsed = 3 * SHORT_LEN + INT_LEN + self.magicLen # Length read in up till now in bytes

            # Figure out number of bytes remaining
            remainingLen = self.headerSize - headerUsed
            if remainingLen > 0:
                # Versions < 4 that supported comments did not have a specific
                # field.  If none of the string flags are set, read a comment.
                if (self.bitMask & (self.USERCOMMENT | self.TIMESTAMP)) > 0:
                    # format specifies included fields

                    # Read user comment if applicable
                    if (self.bitMask & self.USERCOMMENT) > 0 :
                        comment_len = int.from_bytes(file.read(2), byteorder = "big")
                        self.comment = str(file.read(comment_len), 'utf-8')
                    else:
                        self.comment = ""

                    # Read base timestamp (UTC ISO8601) if applicable
                    if (self.bitMask & self.TIMESTAMP) > 0:
                        timestamp_len = int.from_bytes(file.read(2), byteorder = "big")
                        timestamp_str = str(file.read(timestamp_len), 'utf-8')

                        # TODO Loading this as a datetime would make more sense
                        self.timestamp = timestamp_str

                else:
                    # No header information for these fields, assume only a comment
                    # for backward compatibility
                    comment_len = int.from_bytes(file.read(2), byteorder = "big")
                    self.comment = str(file.read(comment_len), 'utf-8')
        else:
            # No Silbido header.
            # Perhaps it can be read in the old headerless format
            self.bitMask = self.DEFAULT
            self.userVersion = -1
            self.version = -1



    def getComment(self):
        '''Return any comment specified by the user'''
        return self.comment

    def getTimestamp(self):
        '''Return any timestamp specified in the annotation'''
        return self.timestamp

    def getUserVersion(self):
        '''Return a version specified by the user'''
        return self.userVersion

    def getFileFormatVersion(self):
        '''Return the version of the file format'''
        return self.version

    def getMask(self):
        '''Return the tonal descriptor mask'''
        return self.bitMask

    def hasScore(self):
        '''Are scores available for each tonal?'''
        return (self.bitMask & self.SCORE) > 0

    def hasConfidence(self):
        '''Are confidences available for each tonal?'''
        return (self.bitMask & self.CONFIDENCE) > 0

    def hasTime(self):
        '''Is time available for each tonal?'''
        return (self.bitMask & self.TIME) > 0

    def hasFreq(self):
        '''Is frequency available for each tonal?'''
        return (self.bitMask & self.FREQ) > 0

    def hasRidge(self):
        '''Is time available for each tonal?'''
        return (self.bitMask & self.RIDGE) > 0

    def hasSNR(self):
        '''Is signal to noise ratio available for each tonal?'''
        return (self.bitMask & self.SNR) > 0

    def hasPhase(self):
        '''Is phase available for each tonal?'''
        return (self.bitMask & self.PHASE) > 0

    def hasSpecies(self):
        '''Is there a species entry for each tonal? (Might be an empty string.)'''
        return (self.bitMask & self.SPECIES) > 0

    def hasCall(self):
        '''Is there a call for each species? (Might be an empty string.)'''
        return (self.bitMask & self.CALL) > 0


class tonalReader:
    def __init__(self, filename):
        '''Creates an itterable object that gets each tonal from a .ann Silbido file'''
        self.file = open(filename, 'rb')

        self.hdr = TonalHeader(self.file)

        self.count = 0

        self.filename = filename
        if self.hdr.userVersion == -1:
            # No header was present, rewind file and use default assumptions
            self.file.close()
            self.file = open(filename, 'rb')

    def __iter__(self):
        return self
    def __len__(self):

        length = len(list(self))
        self.file.close()
        self.__init__(self.filename)

        return length

    def getHeader(self):
        '''Returns the header object for the loaded file'''
        return self.hdr

    def refresh(self):
        '''Resets the file pointer to the first tonal'''
        self.file.close()
        self.__init__(self.filename)


    def getTimeFrequencyContours(self):
        '''Given the current state of the file pointer, returns contours for
        all succeeding contours in the form of a list of lists of
        (time, frequency) tuples.'''

        return [[(n["time"], n["freq"]) for n in tonal["tfnodes"]] for tonal in self]

    def __next__(self):
        '''Returns the next tonal (i.e. the next contour) along with a graph Id in a tuple, (tonal, id)'''

        if len(self.file.peek()) == 0:
            raise StopIteration
        confidence = 0.0
        score = 0.0
        call = ""
        species = ""
        graphId = -1


        self.count = self.count + 1
        time = None
        freq = None
        phase = None
        snr = None
        ridge = False

        # Read in metadata about this tonal if specified
        if self.hdr.hasConfidence():
            confidence = struct.unpack('>d',self.file.read(DOUBLE_LEN))[0]
        if self.hdr.hasScore():
            score = struct.unpack('>d',self.file.read(DOUBLE_LEN))[0]
        if self.hdr.hasSpecies():
            str_len = int.from_bytes(self.file.read(2), byteorder = "big")
            species = str(self.file.read(str_len), 'utf-8')
        if self.hdr.hasCall():
            str_len = int.from_bytes(self.file.read(2), byteorder = "big")
            call = str(self.file.read(str_len), 'utf-8')


        # Read tonal itself
        if self.hdr.getFileFormatVersion() > 2:
            graphId = int.from_bytes(self.file.read(LONG_LEN), byteorder = "big")

        # Read current tonal
        N = int.from_bytes(self.file.read(INT_LEN), byteorder = "big")
        tfnodes = []
        for _ in range(N):
            if (self.hdr.bitMask & self.hdr.TIME) != 0:
                time = struct.unpack('>d',self.file.read(DOUBLE_LEN))[0]
            if (self.hdr.bitMask & self.hdr.FREQ) != 0:
                freq = struct.unpack('>d',self.file.read(DOUBLE_LEN))[0]

            if (self.hdr.bitMask & self.hdr.SNR) != 0:
                snr = struct.unpack('>d',self.file.read(DOUBLE_LEN))[0]
            if (self.hdr.bitMask & self.hdr.PHASE) != 0:
                phase = struct.unpack('>d',self.file.read(DOUBLE_LEN))[0]
            if (self.hdr.bitMask & self.hdr.RIDGE) != 0:
                ridge = struct.unpack('>d',self.file.read(DOUBLE_LEN))[0]

            tfnode = {
                "time": time,
                "freq": freq,
                "snr": snr,
                "phase": phase,
                "ridge": ridge
            }

            tfnodes.append(tfnode)

        return {
            "species": species,
            "call": call,
            "graphId": graphId,
            "confidence": confidence,
            "score": score,
            "tfnodes": tfnodes,
        }

