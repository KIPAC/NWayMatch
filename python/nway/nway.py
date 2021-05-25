""" Proof of concept for n-way matching using footprint detection

Some terminology:

cell    :   A small area, used to find matches,  The size
    should be about the same as the maximum match radius.

source   :  As per DM, per-catalog detection

matchWcs :  The WCS used to define the cells

subRegion : A square sub-region of the skymap defined by the matchWcs

sourceCountsMap :  A map of the number of source per cell, made per
   subRegion

Cluster:  A set of sources found by running a footprint finding algorithm
   on a sourceCountsMap

"""

import sys
import os
import glob

from collections import OrderedDict

import time
import numpy as np
from astropy import wcs
from astropy.table import Table
from astropy.table import vstack
from astropy.io import fits

try:
    import pyarrow.parquet as pq  # noqa
except ImportError:
    print("nway requires pyarrrow")

try:
    import lsst.afw.detection as afwDetect
    import lsst.afw.image as afwImage
except ImportError:
    print("nway requires lsst.afw")    
    
RECURSE_MAX = 200
COLUMNS = ['ra', 'decl', 'visit', 'ccd', 'sky_source', 'sourceId', 'PsFlux', 'PsFluxErr', 'Centroid_flag', 'detect_isPrimary']

def createGlobalWcs(refDir, cellSize, nCell):
    """ Helper function to create the WCS used to project the
    sources in a skymap """
    w = wcs.WCS(naxis=2)
    w.wcs.cdelt = [-cellSize, cellSize]
    w.wcs.crpix = [nCell[0]/2, nCell[1]/2]
    w.wcs.crval = [refDir[0], refDir[1]]
    return w

def clusterStats(clusterDict):
    """ Helper function to get stats about the clusters

    'Orphan'   means single source clusters (i.e., single detections)
    'Mixed`    means there is more that one source from at least one
               input catalog
    'Confused' means there are more than four cases of duplication
    """
    nOrphan = 0
    nMixed = 0
    nConfused = 0
    for val in clusterDict.values():
        if val.nSrc == 1:
            nOrphan += 1
        if val.nSrc != val.nUnique:
            nMixed += 1
            if val.nSrc > val.nUnique + 3:
                nConfused += 1
    return np.array([len(clusterDict), nOrphan, nMixed, nConfused])


class ClusterData:
    """ Class to store data about clusters

    Parameters
    ----------
    iCluster : `int`
        Cluster ID
    origCluster : `int`
        Id of the original cluster this cluster was made from
    nSrc : `int`
        Number of sources in this cluster
    nUnique : `int`
        Number of catalogs contributing sources to this cluster
    catIndices : `np.array`, [`int`]
        Indices of the catalogs of sources associated to this cluster
    sourcdIds : `np.array`, [`int`]
        Sources IDs of the sources associated to this cluster
    sourcdIdxs : `np.array`, [`int`]
        Indices of the sources with their respective catalogs
    xCent : `float`
        X-pixel value of cluster centroid (in WCS used to do matching)
    yCent : `float`
        Y-pixel value of cluster centroid (in WCS used to do matching)
    """
    def __init__(self, iCluster, footprint, sources, origCluster=None):
        self._iCluster = iCluster
        self._footprint = footprint
        if origCluster is None:
            self._origCluster = self._iCluster
        else:
            self._origCluster = origCluster
        self._catIndices = sources[0]
        self._sourceIds = sources[1]
        self._sourceIdxs = sources[2]
        self._nSrc =  self._catIndices.size
        self._nUnique = len(np.unique(self._catIndices))
        self._objects = []
        self._xCent = None
        self._yCent = None
        self._dist2 = None
        self._rmsDist = None
        self.xCell = None
        self.yCell = None
        self.snr = None

    def extract(self, subRegionData):
        """ Extract the xCell, yCell and snr data from
        the sources in this cluster
        """
        self.xCell = np.zeros((self._nSrc), np.float32)
        self.yCell = np.zeros((self._nSrc), np.float32)
        self.snr = np.zeros((self._nSrc), np.float32)
        for i, (iCat, srcIdx) in enumerate(zip(self._catIndices, self._sourceIdxs)):
            self.xCell[i] = subRegionData.data[iCat]['xcell'].values[srcIdx]
            self.yCell[i] = subRegionData.data[iCat]['ycell'].values[srcIdx]
            self.snr[i] = subRegionData.data[iCat]['SNR'].values[srcIdx]

    def clearTempData(self):
        """ Remove temporary data only used when making objects """
        self.xCell = None
        self.yCell = None
        self.snr = None

    @property
    def iCluster(self):
        """ Return the cluster ID """
        return self._iCluster

    @property
    def nSrc(self):
        """ Return the number of sources associated to the cluster """
        return self._nSrc

    @property
    def nUnique(self):
        """ Return the number of catalogs contributing sources to the cluster """
        return self._nUnique

    @property
    def sourceIds(self):
        """ Return the source IDs associated to this cluster """
        return self._sourceIds

    @property
    def dist2(self):
        """ Return an array with the distance squared (in cells)
        between each source and the cluster centroid """
        return self._dist2

    @property
    def objects(self):
        """ Return the objects associated with this cluster """
        return self._objects

    def processCluster(self, subRegionData, pixelR2Cut):
        """ Function that is called recursively to
        split clusters until they:

        1.  Consist only of sources with the match radius of the cluster
        centroid.

        2.  Have at most one source per input catalog
        """
        self._nSrc =  self._catIndices.size
        self._nUnique = len(np.unique(self._catIndices))
        if self._nSrc == 0:
            print("Empty cluster", self._nSrc, self._nUnique)
            return self._objects
        self.extract(subRegionData)
        if self._nSrc == 1:
            self._xCent = self.xCell[0]
            self._yCent = self.yCell[0]
            self._dist2 = np.zeros((1))
            self._rmsDist = 0.
            initialObject = self.addObject(subRegionData)
            initialObject.processObject(subRegionData, pixelR2Cut)
            self.clearTempData()
            return self._objects

        sumSnr = np.sum(self.snr)
        self._xCent = np.sum(self.xCell*self.snr) / sumSnr
        self._yCent = np.sum(self.yCell*self.snr) / sumSnr
        self._dist2 = (self._xCent - self.xCell)**2 + (self._yCent - self.yCell)**2
        self._rmsDist = np.sqrt(np.mean(self._dist2))
        
        initialObject = self.addObject(subRegionData)
        initialObject.processObject(subRegionData, pixelR2Cut)
        self.clearTempData()
        return self._objects

    def addObject(self, subRegionData, mask=None):
        """ Add a new object to this cluster """
        newObject = subRegionData.addObject(self, mask)
        self._objects.append(newObject)
        return newObject


class ObjectData:
    """ Small class to define 'Objects', i.e., sets of associated sources """

    def __init__(self, cluster, objectId, mask):
        """ Build from `ClusterData`, an objectId and mask specifying with sources
        in the cluster are part of the object """
        self._parentCluster = cluster
        self._objectId = objectId
        if mask is None:
            self._mask = np.ones((self._parentCluster.nSrc), dtype=bool)
        else:
            self._mask = mask
        self._catIndices = self._parentCluster._catIndices[self._mask]
        self._nSrc = self._catIndices.size
        self._nUnique = np.unique(self._catIndices).size
        self._xCent = None
        self._yCent = None
        self._dist2 = None
        self._rmsDist = None

    @property
    def nSrc(self):
        """ Return the number of sources associated to the cluster """
        return self._nSrc

    @property
    def nUnique(self):
        """ Return the number of catalogs contributing sources to the cluster """
        return self._nUnique

    @property
    def dist2(self):
        """ Return an array with the distance squared (in cells)
        between each source and the cluster centroid """
        return self._dist2

    def updateCatIndices(self):
        self._catIndices = self._parentCluster._catIndices[self._mask]
        self._nSrc = self._catIndices.size
        self._nUnique = np.unique(self._catIndices).size

    def sourceIds(self):
        return self._parentCluster.sourceIds[self._mask]
        
    def processObject(self, subRegionData, pixelR2Cut, recurse=0):
        """ Recursively process an object and make sub-objects """
        if recurse > RECURSE_MAX:
            print("Recursion limit: ", self._nSrc, self._nUnique)
            return
        if self._nSrc == 0:
            print("Empty object", self._nSrc, self._nUnique, recurse)
            return

        xCell = self._parentCluster.xCell[self._mask]
        yCell = self._parentCluster.yCell[self._mask]
        snr = self._parentCluster.snr[self._mask]

        if self._mask.sum() == 1:
            self._xCent = xCell[0]
            self._yCent = yCell[0]
            self._dist2 = np.zeros((1), float)
            self._rmsDist = 0.
            return

        sumSnr = np.sum(snr)
        self._xCent = np.sum(xCell*snr) / sumSnr
        self._yCent = np.sum(yCell*snr) / sumSnr
        self._dist2 = np.array((self._xCent - xCell)**2 + (self._yCent - yCell)**2)
        self._rmsDist = np.sqrt(np.mean(self._dist2))
        subMask = self._dist2 < pixelR2Cut
        if subMask.all():
            if self._nSrc != self._nUnique:
                self.splitObject(subRegionData, pixelR2Cut, recurse=recurse+1)
            return

        if not subMask.any():
            idx = np.argmax(snr)
            self._xCent = xCell[idx]
            self._yCent = yCell[idx]
            self._dist2 = np.array((self._xCent - xCell)**2 + (self._yCent - yCell)**2)
            self._rmsDist = np.sqrt(np.mean(self._dist2))
            subMask = self._dist2 < pixelR2Cut

        newObjMask = self._mask.copy()
        newObjMask[newObjMask] *= subMask

        newObject = self._parentCluster.addObject(subRegionData, newObjMask)
        newObject.processObject(subRegionData, pixelR2Cut)

        self._mask[self._mask] *= ~subMask
        self.updateCatIndices()
        self.processObject(subRegionData, pixelR2Cut, recurse=recurse+1)


    def splitObject(self, subRegionData, pixelR2Cut, recurse=0):
        """ Split up a cluster keeping only one source per input
        catalog, choosing the one closest to the cluster center """
        sortIdx = np.argsort(self._dist2)
        mask = np.ones((self._nSrc), dtype=bool)
        usedCats = {}
        for iSrc, catIdx in zip(sortIdx, self._catIndices[sortIdx]):
            if catIdx not in usedCats:
                usedCats[catIdx] = 1
                continue
            else:
                usedCats[catIdx] += 1
            mask[iSrc] = False

        newObjMask = self._mask.copy()
        newObjMask[newObjMask] *= mask

        newObject = self._parentCluster.addObject(subRegionData, newObjMask)
        newObject.processObject(subRegionData, pixelR2Cut)

        self._mask[self._mask] *= ~mask
        self.updateCatIndices()        
        self.processObject(subRegionData, pixelR2Cut, recurse=recurse+1)


class SubregionData:
    """ Class to analyze data for a SubRegion

    Include sub-region boundries, reduced data tables
    and clustering results

    Does not store sky maps

    Subregions are square sub-regions of the Skymap
    constructed with the WCS

    The subregion covers corner:corner+size

    The sources are projected into an array that extends `buf` cells
    beyond the region.

    Parameters
    ----------
    _data : `list`, [`Dataframe`]
        Reduced dataframes with only sources for this sub-region

    _clusterIds : `list`, [`np.array`]
        Matched arrays with the index of the cluster associated to each
        source.  I.e., these could added to the Dataframes as
        additional columns

    _clusterDict : `dict`, [`int` : `ClusterData`]
        Dictionary with cluster membership data

    TODO:  Add code to filter out clusters centered in the buffer
    """
    def __init__(self, matcher, idOffset, corner, size, buf=10):
        self._matcher = matcher
        self._idOffset = idOffset # Offset used for the Object and Cluster IDs for this region
        self._corner = corner # cellX, cellY for corner of region
        self._size = size # size of region
        self._buf = buf
        self._minCell = corner - buf
        self._maxCell = corner + size + buf
        self._nCells = self._maxCell - self._minCell
        self._data = None
        self._nSrc = None
        self._footprintIds = None
        self._clusterDict = OrderedDict()
        self._objectDict = OrderedDict()

    def reduceData(self, data):
        """ Pull out only the data needed for this sub-region """
        self._data = [self.reduceDataframe(val) for val in data]
        self._nSrc = sum([len(df) for df in self._data])
        
    @property
    def nClusters(self):
        """ Return the number of clusters in this region """
        return len(self._clusterDict)

    @property
    def nObjects(self):
        """ Return the number of objects in this region """
        return len(self._objectDict)

    @property
    def data(self):
        """ Return the data associated to this region """
        return self._data

    @property
    def clusterDist(self):
        """ Return a dictionary mapping clusters Ids to clusters """
        return self._clusterDict

    def reduceDataframe(self, dataframe):
        """ Filters dataframe to keep only source in the subregion """
        xLocal = dataframe['xcell'] - self._minCell[0]
        yLocal = dataframe['ycell'] - self._minCell[1]
        filtered = (xLocal >= 0) & (xLocal < self._nCells[0]) & (yLocal >= 0) & (yLocal < self._nCells[1])
        red = dataframe[filtered].copy(deep=True)
        red['xlocal'] = xLocal[filtered]
        red['ylocal'] = yLocal[filtered]
        return red

    def countsMap(self, weightName=None):
        """ Fill a map that counts the number of source per cell """
        toFill = np.zeros((self._nCells))
        for df in self._data:
            toFill += self.fillSubRegionFromDf(df, weightName=weightName)
        return toFill

    def associateSourcesToFootprints(self, clusterKey):
        """ Loop through data and associate sources to clusters """
        self._footprintIds = [self.findClusterIds(df, clusterKey) for df in self._data]

    def buildClusterData(self, fpSet, pixelR2Cut=4.):
        """ Loop through cluster ids and collect sources into
        the ClusterData objects """
        footprints = fpSet.getFootprints()
        footprintDict = {}
        nMissing = 0
        nFound = 0
        for iCat, (df, footprintIds) in enumerate(zip(self._data, self._footprintIds)):
            for srcIdx, (srcId, footprintId) in enumerate(zip(df['sourceId'], footprintIds)):
                if footprintId < 0:
                    nMissing += 1
                    continue
                if footprintId not in footprintDict:
                    footprintDict[footprintId] = [(iCat, srcId, srcIdx)]
                else:
                    footprintDict[footprintId].append((iCat, srcId, srcIdx))
                nFound += 1
        for footprintId, sources in footprintDict.items():
            footprint = footprints[footprintId]
            iCluster = footprintId+self._idOffset
            cluster = ClusterData(iCluster, footprint, np.array(sources).T)
            self._clusterDict[iCluster] = cluster
            cluster.processCluster(self, pixelR2Cut)

    def analyze(self, weightName=None, pixelR2Cut=4.):
        """ Analyze this sub-region

        Note that this returns the counts maps and clustering info,
        which can be helpful for debugging.
        """
        if self._nSrc == 0:
            return None
        countsMap = self.countsMap(weightName)
        oDict = self.getFootprints(countsMap)
        oDict['countsMap'] = countsMap
        self.associateSourcesToFootprints(oDict['footprintKey'])
        self.buildClusterData(oDict['footprints'], pixelR2Cut)
        return oDict

    @staticmethod
    def findClusterIds(df, clusterKey):
        """ Associate sources to clusters using `clusterkey`
        which is a map where any pixel associated to a cluster
        has the cluster index as its value """
        return np.array([clusterKey[yLocal,xLocal] for xLocal, yLocal in zip(df['xlocal'], df['ylocal'])]).astype(np.int32)

    def fillSubRegionFromDf(self, df, weightName=None):
        """ Fill a source counts map from a reduced dataframe for one input
        catalog """
        if weightName is None:
            weights = None
        else:
            weights = df[weightName].values
        hist = np.histogram2d(df['xlocal'], df['ylocal'], bins=self._nCells,
                              range=((0, self._nCells[0]),
                                     (0, self._nCells[1])),
                              weights=weights)
        return hist[0]

    @staticmethod
    def filterFootprints(fpSet, buf):
        """ Remove footprints within `buf` cells of the region edge """
        region = fpSet.getRegion()
        width, height = region.getWidth(), region.getHeight()
        outList = []
        maxX = width - buf
        maxY = height - buf
        for fp in fpSet.getFootprints():
            cent = fp.getCentroid()
            xC = cent.getX()
            yC = cent.getY()
            if xC < buf or xC > maxX or yC < buf or yC > maxY:
                continue
            outList.append(fp)
        fpSetOut = afwDetect.FootprintSet(fpSet.getRegion())
        fpSetOut.setFootprints(outList)
        return fpSetOut

    def getFootprints(self, countsMap):
        """ Take a source counts map and do clustering using Footprint detection
        """
        image = afwImage.ImageF(countsMap.astype(np.float32))
        footprintsOrig = afwDetect.FootprintSet(image, afwDetect.Threshold(0.5))
        footprints = self.filterFootprints(footprintsOrig, self._buf)
        footprintKey = afwImage.ImageI(np.full(countsMap.shape, -1, dtype=np.int32))
        for i, footprint in enumerate(footprints.getFootprints()):
            footprint.spans.setImage(footprintKey, i, doClip=True)
        return dict(image=image, footprints=footprints, footprintKey=footprintKey)

    def getClusterAssociations(self):
        """ Convert the clusters to a set of associations """
        clusterIds = []
        sourceIds = []
        distances = []
        for cluster in self._clusterDict.values():
            clusterIds.append(np.full((cluster.nSrc), cluster.iCluster, dtype=int))
            sourceIds.append(cluster.sourceIds)
            distances.append(cluster.dist2)
        if not distances:
            return Table(dict(distance=[], id=np.array([], int), object=np.array([], int)))
        distances = np.hstack(distances)
        distances = self._matcher.cellToArcsec() * np.sqrt(distances)
        data = dict(object=np.hstack(clusterIds),
                    id=np.hstack(sourceIds),
                    distance=distances)
        return Table(data)

    def getObjectAssociations(self):
        clusterIds = []
        objectIds = []
        sourceIds = []
        distances = []
        for obj in self._objectDict.values():
            clusterIds.append(np.full((obj._nSrc), obj._parentCluster.iCluster, dtype=int))
            objectIds.append(np.full((obj._nSrc), obj._objectId, dtype=int))
            sourceIds.append(obj.sourceIds())
            distances.append(obj.dist2)
        if not distances:
            return Table(dict(object=np.array([], int),
                              parent=np.array([], int),
                              id=np.array([], int),
                              distance=[]))
        distances = np.hstack(distances)
        distances = self._matcher.cellToArcsec() * np.sqrt(distances)            
        data = dict(object=np.hstack(objectIds),
                    parent=np.hstack(clusterIds),
                    id=np.hstack(sourceIds),
                    distance=distances)
        return Table(data)

    def getClusterStats(self):
        """ Convert the clusters to a set of associations """
        nClust = self.nClusters
        clusterIds = np.zeros((nClust), dtype=int)
        nSrcs = np.zeros((nClust), dtype=int)
        nObjects = np.zeros((nClust), dtype=int)
        nUniques = np.zeros((nClust), dtype=int)
        distRms = np.zeros((nClust), dtype=float)
        xCents = np.zeros((nClust), dtype=float)
        yCents = np.zeros((nClust), dtype=float)
        for idx, cluster in enumerate(self._clusterDict.values()):
            clusterIds[idx] = cluster._iCluster
            nSrcs[idx] = cluster.nSrc
            nObjects[idx] = len(cluster._objects)
            nUniques[idx] = cluster.nUnique
            distRms[idx] = cluster._rmsDist
            xCents[idx] = cluster._xCent
            yCents[idx] = cluster._yCent
        ra, decl = self._matcher.cellToWorld(xCents, yCents)
        distRms *= self._matcher.cellToArcsec()

        data = dict(clusterIds=clusterIds,
                    nSrcs=nSrcs,
                    nObject=nObjects,
                    nUnique=nUniques,
                    distRms=distRms,
                    ra=ra,
                    decl=decl)

        return Table(data)

    def getObjectStats(self):
        """ Convert the clusters to a set of associations """
        nObj = self.nObjects
        clusterIds = np.zeros((nObj), dtype=int)
        objectIds = np.zeros((nObj), dtype=int)
        nSrcs = np.zeros((nObj), dtype=int)
        distRms = np.zeros((nObj), dtype=float)
        xCents = np.zeros((nObj), dtype=float)
        yCents = np.zeros((nObj), dtype=float)
        for idx, obj in enumerate(self._objectDict.values()):
            clusterIds[idx] = obj._parentCluster._iCluster
            objectIds[idx] = obj._objectId
            nSrcs[idx] = obj.nSrc
            distRms[idx] = obj._rmsDist
            xCents[idx] = obj._xCent
            yCents[idx] = obj._yCent

        ra, decl = self._matcher.cellToWorld(xCents, yCents)
        distRms *= self._matcher.cellToArcsec()
        
        data = dict(clusterIds=clusterIds,
                    objectIds=objectIds,
                    nSrcs=nSrcs,
                    distRms=distRms,
                    ra=ra,
                    decl=decl)

        return Table(data)

    def addObject(self, cluster, mask=None):
        """ Add an object to this sub-region """
        objectId = self.nObjects + self._idOffset
        newObject = ObjectData(cluster, objectId, mask)
        self._objectDict[objectId] = newObject
        return newObject


class NWayMatch:
    """ Class to do N-way matching

    Uses a provided WCS to define a Skymap that covers the full region
    begin matched.

    Uses that WCS to assign cell locations to all sources in the input catalogs

    Iterates over sub-regions and does source clustering in each sub-region
    using Footprint detection on a Skymap of source counts per cell.

    Assigns each input source to a cluster.

    At that stage the clusters are not the final product as they can include
    more than one soruce from a given catalog.

    Loops over clusters and processes each cluster to:

       1. Remove outliers outside the match radius w.r.t. the cluster centroid.
       2. Resolve cases of confusion, where multiple sources from a single
       catalog contribute to a cluster.

    Parameters
    ----------
    _redData : `list`, [`Dataframe`]
        Reduced dataframes with only the columns needed for matching

    _clusters : `OrderedDict`, [`tuple`, `SubregionData`]
        Dictionary providing access to subregion data
    """

    def __init__(self, matchWcs, **kwargs):
        self._wcs = matchWcs
        self._cellSize = self._wcs.wcs.cdelt[1]
        self._nCellSide = np.ceil(2*np.array(self._wcs.wcs.crpix)).astype(int)
        self._subRegionSize = kwargs.get('subRegionSize', 3000)
        self._subRegionBuffer = kwargs.get('subRegionBuffer', 10)
        self._subregionMaxObject = kwargs.get('subregionMaxObject', 100000)
        self._pixelR2Cut = kwargs.get('pixelR2Cut', 1.0)
        self._nSubRegion = np.ceil(self._nCellSide/self._subRegionSize)
        self._redData = OrderedDict()
        self._clusters = None

    def cellToArcsec(self):
        return 3600. * self._cellSize

    def cellToWorld(self, xCell, yCell):
        return self._wcs.wcs_pix2world(xCell, yCell, 0)
    
    @classmethod
    def create(cls, refDir, regionSize, cellSize, **kwargs):
        """ Make an `NWayMatch` object from inputs """
        nCell = (np.array(regionSize)/cellSize).astype(int)
        matchWcs = createGlobalWcs(refDir, cellSize, nCell)
        return cls(matchWcs, **kwargs)

    @property
    def redData(self):
        """ Return the dictionary of reduced data, i.e., just the columns
        need for matching """
        return self._redData

    @property
    def nSubRegion(self):
        """ Return the number of sub-regions in X,Y """
        return self._nSubRegion

    def reduceData(self, inputFiles, visitIds):
        """ Read input files and filter out only the columns we need """
        for fName, vid in zip(inputFiles, visitIds):
            self._redData[vid] = self.reduceDataFrame(fName)

    def reduceDataFrame(self, fName):
        """ Read and reduce a single input file """
        parq = pq.read_pandas(fName, columns=COLUMNS)
        df = parq.to_pandas()
        df['SNR'] = df['PsFlux']/df['PsFluxErr']
        # select sources that have SNR > 5.
        # You may start with 10 or even 50 if you want to start with just the brightest objects
        # AND
        # Centroid_flag is True if there was a problem fitting the position (centroid)
        # AND
        # sky_source is True if it is a measurement of blank sky.
        # sky_sources should have SNR < 5 or the Centroid_flag set,
        # but explicitly filter just to make sure.
        # AND
        # detect_isPrimary = True to remove duplicate rows from deblending:
        # If a source has been deblended, the parent is marked detect_isPrimary=False and its children True.
        df_clean = df[(df.SNR > 5) & ~df.Centroid_flag & ~df.sky_source & df.detect_isPrimary]
        xcell, ycell = self._wcs.wcs_world2pix(df_clean['ra'].values, df_clean['decl'].values, 0)
        df_red = df_clean[["ra", "decl", "SNR", "sourceId"]].copy(deep=True)
        df_red['xcell'] = xcell
        df_red['ycell'] = ycell
        return df_red[["ra", "decl", "SNR", "sourceId", "xcell", "ycell"]]

    def reduceCatalog(self, catalog):
        """ Reduce a catalog """
        raise NotImplementedError()

    def add(self, catalog, vid):
        """ Add a catalog to the data set being matched """
        self._redData[vid] = self.reduceCatalog(catalog)

    def getIdOffset(self, ix, iy):
        """ Get the ID offset to use for a given sub-region """
        subRegionIdx = self._nSubRegion[1]*ix + iy
        return int(self._subregionMaxObject * subRegionIdx)

    def analyzeSubregion(self, ix, iy, fullData=False):
        """ Analyze a single subregion

        Returns an OrderedDict

        'srd' : `SubregionData`
            The analysis data for the sub-region

        if fullData is True the return dict will include

        'image' : `afwImage.ImageI`
            Image of subregion source counts map
        'countsMap' : `np.array`
            Numpy array with same
        'clusters' : `afwDetect.FootprintSet`
            Clusters as dectected by finding FootprintSet on source counts map
        'clusterKey' : `afwImage.ImageI`
            Map of subregion with pixels filled with index of
            associated Footprints
        """
        iSubRegion = np.array([ix, iy])
        corner = iSubRegion * self._subRegionSize
        idOffset = self.getIdOffset(ix, iy)
        srd = SubregionData(self, idOffset, corner, self._subRegionSize, self._subRegionBuffer)
        srd.reduceData(self._redData.values())
        oDict = srd.analyze(pixelR2Cut=self._pixelR2Cut)
        if oDict is None:
            return None
        if fullData:
            oDict['srd'] = srd
            return oDict
        if srd.nObjects >= self._subregionMaxObject:
            print("Too many object in a subregion", srd.nObjects, elf._subregionMaxObject)
        return dict(srd=srd)

    def finish(self):
        """ Does clusering for all subregions

        Does not store source counts maps for the counts regions
        """
        self._clusters = OrderedDict()
        nAssoc = 0
        clusterAssocTables = []
        objectAssocTables = []
        clusterStatsTables = []
        objectStatsTables = []

        for ix in range(int(self._nSubRegion[0])):
            sys.stdout.write("%2i " % ix)
            sys.stdout.flush()
            for iy in range(int(self._nSubRegion[1])):
                sys.stdout.write('.')
                sys.stdout.flush()
                iSubRegion = (ix, iy)
                odict = self.analyzeSubregion(ix, iy)
                if odict is None:
                    continue
                subregionData = odict['srd']
                self._clusters[iSubRegion] = subregionData
                clusterAssocTables.append(subregionData.getClusterAssociations())
                objectAssocTables.append(subregionData.getObjectAssociations())
                clusterStatsTables.append(subregionData.getClusterStats())
                objectStatsTables.append(subregionData.getObjectStats())
                
            sys.stdout.write('!\n')

        sys.stdout.write("Making association vectors\n")
        hduList = fits.HDUList([fits.PrimaryHDU(),
                                fits.table_to_hdu(vstack(clusterAssocTables)),
                                fits.table_to_hdu(vstack(objectAssocTables)),
                                fits.table_to_hdu(vstack(clusterStatsTables)),
                                fits.table_to_hdu(vstack(objectStatsTables))])
        return hduList

    def allStats(self):
        """ Helper function to print info about clusters """
        stats = np.zeros((4), int)
        for key, srd in self._clusters.items():
            subRegionStats = clusterStats(srd._clusterDict)
            print("%3i, %3i: %8i %8i %8i %8i" % (key[0], key[1], subRegionStats[0], subRegionStats[1], subRegionStats[2], subRegionStats[3]))
            stats += subRegionStats
        return stats

def main():
    """ Example usage """

    DATADIR = "."
    SOURCE_TABLEFILES = glob.glob(os.path.join(DATADIR, "sourceTable-*.parq"))
    VISIT_IDS = np.arange(len(SOURCE_TABLEFILES))

    REF_DIR = (150., 2.)  # RA, DEC in deg
    REGION_SIZE = (3., 3.)  # in Deg
    #CELL_SIZE = 5.0e-5    # in Deg
    CELL_SIZE = 1. / (3600*2) # in Deg
    #SUBREGION_SIZE = 2700 # in Pixels
    SUBREGION_SIZE = 1350 # in Pixels
    PIXEL_R2CUT = 1.
    
    t0 = time.time()
    nWay = NWayMatch.create(REF_DIR, REGION_SIZE, CELL_SIZE, pixelR2Cut=PIXEL_R2CUT, subRegionSize=SUBREGION_SIZE)
    print("Building clusters in %ix%i sub-regions" % (nWay.nSubRegion[0], nWay.nSubRegion[1]))
    nWay.reduceData(SOURCE_TABLEFILES, VISIT_IDS)
    outTables = nWay.finish()
    t1 = time.time()
    print("Reading and clustering took %s s" % (t1-t0))

    print("Cluster Summaries for sub-regions")
    print("Region  :  nCluster nOrphan  nMixed   nConf")
    stats = nWay.allStats()
    print("Total:   %8i %8i %8i %8i" % (stats[0], stats[1], stats[2], stats[3]))

    outTables.writeto("out.fits", overwrite=True)

if __name__ == '__main__':
    main()
