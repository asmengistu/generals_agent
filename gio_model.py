from collections import defaultdict

from constants import GioConstants

import gzip
import json
import numpy as np

# Armies, Owners, Types
ARMY_CH = 0
OWNER_CH = 1
TYPE_CH = 2

MOORE_NEIGHBORHOOD = np.array([
  (-1, -1), (-1, 0), (-1, 1),
  (0, -1), (0, 0), (0, 1),
  (1, -1), (1, 0), (1, 1)
])


class GioModel(object):
  """Class used to represent a replay of a GeneralsIO game."""
  def __init__(self, model_string):
    model = json.loads(model_string)
    self.model_ = model
    self.buildBoard_(model)
    self._buildfogOfWarLookup()

  @staticmethod
  def fromFile(filename):
    """Loads a gio model from a file. Supports loading from gzipped files."""
    opener = gzip.open if '.gz' in filename else open
    with opener(filename, 'r') as f:
      contents = f.read()
    return GioModel(contents)

  def buildBoard_(self, model):
    self.width = model['height']
    self.height = model['width']
    width, height = self.width, self.height
    reshapeFn = lambda l: np.reshape(l, (width, height))
    self.num_players = model['num_players']
    self.num_turns = len(model['types'])
    self.id = model['id']
    self.usernames = model['usernames']
    self.stars = model['stars']
    self.afks = model['afks']
    self.ranks = model['ranks']

    self.board_ = np.zeros((self.getNumTurns(),
                           width,
                           height,
                           GioConstants.num_channels))
    board = self.board_
    board[0, :, :, ARMY_CH] = reshapeFn(model['armies'][0])
    board[0, :, :, OWNER_CH] = reshapeFn(model['owners'][0])
    board[0, :, :, TYPE_CH] = reshapeFn(model['types'][0])
    for turn in xrange(1, self.getNumTurns()):
      # If the values didn't change after a turn, the arrays will be empty.
      if not model['armies'][turn]:
        board[turn, :, :, ARMY_CH] = board[turn - 1, :, :, ARMY_CH]
      else:
        board[turn, :, :, ARMY_CH] = reshapeFn(model['armies'][turn])
      if not model['owners'][turn]:
        board[turn, :, :, OWNER_CH] = board[turn - 1, :, :, OWNER_CH]
      else:
        board[turn, :, :, OWNER_CH] = reshapeFn(model['owners'][turn])
      if not model['types'][turn]:
        board[turn, :, :, TYPE_CH] = board[turn - 1, :, :, TYPE_CH]
      else:
        board[turn, :, :, TYPE_CH] = reshapeFn(model['types'][turn])

  def getId(self):
    return self.id

  def _buildfogOfWarLookup(self):
    self._neighbors = defaultdict(list)
    width, height = self.getMapSize()
    for i in xrange(width):
      for j in xrange(height):
        neighbors = MOORE_NEIGHBORHOOD + (i, j)
        for x, y in neighbors:
          if x >= 0 and y >= 0 and x < width and y < height:
            self._neighbors[(i, j)].append((x, y))

  def isVisible(self, turn, i, j, player):
    """Returns whether square i, j is visible to `player` at `turn`."""
    for (x, y) in self._neighbors[(i, j)]:
      if self.board_[turn, x, y, OWNER_CH] == player:
        return True
    return False

  def clipBoard(self, turn):
    """Trims the board to [:turn, :, :, :]"""
    self.board_ = self.board_[:turn, :, :, :]
    self.num_turns = turn

  def getBoardView(self, player):
    """Returns a 1st persion view from `player`s POV. Fog of war is type 0."""
    new_board = np.zeros(self.board_.shape)
    board = self.board_
    width, height = self.getMapSize()
    for turn in xrange(self.num_turns):
      for i in xrange(width):
        for j in xrange(height):
          if self.isVisible(turn, i, j, player):
            new_board[turn, i, j, ARMY_CH] = board[turn, i, j, ARMY_CH]
            # Owner is always 0.
            owner = board[turn, i, j, OWNER_CH]
            # Switch player index to 0.
            if player != 0 and owner == 0:
              new_board[turn, i, j, OWNER_CH] = player
            elif player != 0 and owner == player:
              new_board[turn, i, j, OWNER_CH] = 0
            else:
              new_board[turn, i, j, OWNER_CH] = owner
            # If visible type is just copied.
            new_board[turn, i, j, TYPE_CH] = board[turn, i, j, TYPE_CH]
          else:
            cell_type = board[turn, i, j, TYPE_CH]
            if cell_type in (2, 3):
              new_board[turn, i, j, TYPE_CH] = 5
            new_board[turn, i, j, OWNER_CH] = -1
    return new_board

  def getScoreBoard(self):
    """Returns (army, fort, land) counts for all players."""
    army = np.zeros((self.num_turns, self.getNumPlayers()))
    forts = np.zeros((self.num_turns, self.getNumPlayers()))
    land = np.zeros((self.num_turns, self.getNumPlayers()))
    width, height = self.getMapSize()
    for turn in xrange(self.num_turns):
      for i in xrange(width):
        for j in xrange(height):
          owner = int(self.board_[turn][i, j, OWNER_CH])
          if owner >= 0:
            land[turn, owner] += 1
            army[turn, owner] += self.board_[turn][i, j, ARMY_CH]
            if self.board_[turn][i, j, TYPE_CH] in (3, 4):
              forts[turn, owner] += 1
    return {
      'army': army,
      'forts': forts,
      'land': land
    }

  def getNumPlayers(self):
    return self.num_players

  def getNumTurns(self):
    return self.num_turns

  def getMapSize(self):
    """(width, height) tuple."""
    return (self.width, self.height)
