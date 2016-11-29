//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include <memory>
#include <set>

#pragma warning(push)
#pragma warning(disable : 4996) // Due to multiple unsafe functions in fileutil.h
#include "ComputationNetwork.h"
#include "DistGradHeader.h"
#include "LinearAlgebraNodes.h"
#include "MPIWrapper.h"
#include "Matrix.h"
#include "SimpleDistGradAggregator.h"
#include "V2SimpleDistGradAggregator.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <typename ElemType>
void AggregateAccumulatorValuesAndUpdateEvaluation(
    std::shared_ptr<ComputationNetwork> net,
    std::set<std::shared_ptr<ComputationNodeBase>> evalNodesWhichAccumulateResult,
    std::shared_ptr<DistGradHeader> gradHeader,
    std::shared_ptr<MPIWrapper> mpi)
{
    // Accumulator stores mean value and number of samples. Aggregation performs simple summation of values,
    // so we transfer sum instead of mean, and calculate mean after aggregation is finished.
    auto allEpochAccumulatorNodes = net->GetNodesWithType(OperationNameOf(EpochAccumulatorNode));
    std::vector<Matrix<ElemType>*> accumulatorValues;
    size_t sampleCount =
        dynamic_pointer_cast<EpochAccumulatorNode<ElemType>>(allEpochAccumulatorNodes.front())->GetNumberOfSamples();
    // Calculate accumulator sums.
    for (auto& accumulatorNode : allEpochAccumulatorNodes)
    {
        auto node = dynamic_pointer_cast<EpochAccumulatorNode<ElemType>>(accumulatorNode);
        assert(sampleCount == node->GetNumberOfSamples());
        Matrix<ElemType>& accumulator = *node->GetAccumulator();
        accumulator *= (ElemType) sampleCount;
        accumulatorValues.emplace_back(&accumulator);
    }

    // Prepare aggregator.
    std::shared_ptr<IDistGradAggregator<ElemType>> distGradAgg;
    if (Globals::UseV2Aggregator())
        distGradAgg = make_shared<V2SimpleDistGradAggregator<ElemType>>(
            mpi,
            false /*useAsyncAggregation*/,
            0 /*syncStatsTrace*/,
            ::CNTK::MPICommunicator());
    else
        distGradAgg = make_shared<SimpleDistGradAggregator<ElemType>>(
            mpi,
            false /*useAsyncAggregation*/,
            net->GetDeviceId(),
            0 /*syncStatsTrace*/);

    // Prepare header.
    const size_t c_evalNodes = 1;
    if (gradHeader == nullptr)
        gradHeader.reset(DistGradHeader::Create(c_evalNodes),
                         [](DistGradHeader* ptr) { DistGradHeader::Destroy(ptr); });
    gradHeader->numEvalNode = c_evalNodes;
    gradHeader->numSamples = sampleCount;
    gradHeader->numSamplesWithLabel = sampleCount;
    gradHeader->criterion = 0.0; // (not used here)
    for (size_t i = 0; i < c_evalNodes; i++)
        // Not used here, but at least one is required by aggregation.
        gradHeader->evalErrors[i] = std::make_pair<double, size_t>(0.0, 0);

    // Aggregate accumulator sums.
    bool samplesProcessed = distGradAgg->AggregateGradients(accumulatorValues, gradHeader.get(), /*resetState =*/false);
    if (!samplesProcessed)
        RuntimeError("Couldn't aggregate accumulator values.");

    // Accumulators should contain mean values. We calculated them based on aggregated sums and number of samples.
    for (Matrix<ElemType>* acc : accumulatorValues)
        (*acc) /= (ElemType) gradHeader->numSamples;

    // Update output values of accumulator nodes.
    for (auto& accumulatorNode : allEpochAccumulatorNodes)
    {
        auto node = dynamic_pointer_cast<EpochAccumulatorNode<ElemType>>(accumulatorNode);
        node->SetNumberOfSamples(gradHeader->numSamples);
        node->BeginForwardProp();
        node->CopyAccumulatorToValue();
        node->EndForwardProp();
        node->BumpEvalTimeStamp();
    }

    // Update output values of nodes between accumulator nodes and evaluation nodes.
    net->ForwardProp(evalNodesWhichAccumulateResult);
}
}}}
#pragma warning(pop)