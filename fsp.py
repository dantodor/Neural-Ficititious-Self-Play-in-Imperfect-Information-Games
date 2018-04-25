'''

    Ficticous self-play



class player:

    MAR = replayBuffer
    MBR = replayBuffer

    M = ReplayBuffer

    M = [
        [s, a, r, s2 = ra],
        [s(i), a(i), ra(i), r],
        [s(i+1), a(i+1), ra(i+1), r]
    ]

    BRNetwork = intiBR
    ARNetwork = initAR

    function intiBR(weights):
        -> intialize nuronal network
        layer1: 3 neurons (mycard, maincard, pot)
        hiddenlayers: ?
        layerOut: 3 neurons (call, fold, raise)

    function initAR(weights):
        -> intialize nuronal network
        layer1: 3 neurons (mycard, maincard, pot)
        hiddenlayers: ?
        layerOut: 3 neurons (call, fold, raise)

    function updateBestResponse:
        RLOptimizer.train(BRNetwork, M) 

    function updateAverageResponse:
        SLOptimizer.train(ARNetwork, M)

class main:
    init 2 players from class player
    env = ENV

    function ficticousSelfPlay(env, player, episodes(n and m)):
        foreach player:
            player.intiBR(arbitraryWeights)
            player.initAR(sameWeights)
        
        ny = mixingParameters()
        Dataset = generateData()

    function mixingParameters:
        
    function generateData:
        strategy = parts from BR and parts AR
        Dataset = foreach n(episode) play with strategy
        foreach player:
            DatasetForSpecificPlayer = play with own BR and opponents AR
            DatasetForSpecificPlayer = Dataset and DatasetForSpecificPlayer
        return DatasetForSpecificPlayer foreach Player

'''


