// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

interface IOrbitalMATH_HELPER {
    function sqrtQ96X48(uint144 y) external pure returns (uint144);

    function getTickParameters(
        uint144 depegLimit,
        uint144 reserve
    ) external pure returns (uint144, uint144);

    function calculateBoundaryTickS(
        uint144 radius,
        uint144 k
    ) external pure returns (uint144);

    function solveTorusInvariant(
        uint144 sumInteriorReserves,
        uint144 interiorConsolidatedRadius,
        uint144 boundaryConsolidatedRadius,
        uint144 boundaryTotalKBound,
        uint144 tokenInIndex,
        uint144 tokenOutIndex,
        uint144 amountInAfterFee,
        uint144[] memory totalReserves
    ) external view returns (uint144);

    function solveQuadraticInvariant(
        uint144 deltaLinear,
        uint144[] memory reserves,
        uint144 tokenInIndex,
        uint144 tokenOutIndex,
        uint144 consolidatedRadius,
        uint144 kCross
    ) external pure returns (uint144);
}

/**
 * @title OrbitalPool
 * @notice A 5-token AMM implementing the Orbital mathematical model
 * @dev Uses a torus-based invariant for price discovery and liquidity management. It uses the notation Q96.48 to represent decimals.
 * @dev Every numerical value is marked in this format unless marked otherwise.
 * @author Orbital Protocol
 */
contract OrbitalPool {
    using SafeERC20 for IERC20;

    uint256 public immutable TOKENS_COUNT; // This is not in Q96.48 format
    uint144 public ROOT_N ;
    IOrbitalMATH_HELPER public immutable MATH_HELPER;
    IERC20[] public tokens;
    uint144[] public totalReserves;
    mapping(uint144 p => Tick) public activeTicks;
    uint144[] public allP;
    
    /**
     * @notice Status of a tick in the pool
     */
    enum TickStatus {
        Interior, /// @dev Tick is in the interior region
        Boundary /// @dev Tick is on the boundary
    }

    /**
     * @notice Represents a liquidity tick in the pool
     * @dev Each tick has its own reserves and LP shares
     */
    struct Tick {
        uint144 p;
        uint144 r; /// @dev Radius of the tick
        uint144 k; /// @dev Plane constant for the tick
        uint144 liquidity; /// @dev Total liquidity in the tick
        uint144 totalLpShares; /// @dev Total LP shares issued
        TickStatus status; /// @dev Current status of the tick
        mapping(address => uint144) lpShares; /// @dev LP shares per address
        uint144[] reserves; /// @dev Reserves of each token
    }

    /**
     * @notice Consolidated data for interior and boundary ticks
     * @dev Used for efficient swap calculations
     */
    struct ConsolidatedTickData {
        uint144 totalLiquidity; /// @dev Combined liquidity
        uint144 tickCount; /// @dev Number of ticks
        uint144 consolidatedRadius; /// @dev Combined radius
        uint144 totalKBound; /// @dev Sum of k values for boundary ticks
        uint144[] totalReserves; /// @dev Sum of reserves across ticks
        uint144[] sumSquaredReserves; /// @dev Sum of squared reserves
    }

    /**
     * @notice Emitted when liquidity is added to a tick
     * @param provider Address of the liquidity provider
     * @param amounts Amounts of each token added
     * @param lpShares LP shares minted
     */
    event LiquidityAdded(
        uint144[] amounts,
        uint144 lpShares,
        address indexed provider,
        uint144 p
    );
    event TickParams(uint144 k, uint144 r);
    /**
     * @notice Emitted when liquidity is removed from a tick
     * @param provider Address of the liquidity provider
     * @param amounts Amounts of each token removed
     */
    event LiquidityRemoved(
        uint144[] amounts,
        address indexed provider,
        uint144 p
    );

    /**
     * @notice Emitted when a swap is executed
     * @param trader Address of the trader
     * @param tokenIn Index of input token
     * @param tokenOut Index of output token
     * @param amountIn Input amount
     * @param amountOut Output amount
     * @param fee Fee charged
     */
    event Swap(
        address indexed trader,
        uint144 tokenIn,
        uint144 tokenOut,
        uint144 amountIn,
        uint144 amountOut,
        uint144 fee
    );

    error InvalidKValue();
    error SameToken();
    error InvalidLength();
    error InvalidAmounts();
    error InsufficientLiquidity();
    error SlippageExceeded();
    error NumericalError();
    error UnsatisfiedInvariant();
    error NoInteriorLiquidity();
    error InvalidReserves();
    error TorusInvariantCallFailed();

    /**
     * @notice Initializes the Orbital pool
     * @param _tokens Array of ERC20 tokens for the pool
     * @param mathHelperAddress Address of the Stylus math helper contract
     */
    constructor(IERC20[] memory _tokens, address mathHelperAddress) {
        require(_tokens.length > 0, "At least one token required");
        MATH_HELPER = IOrbitalMATH_HELPER(mathHelperAddress);
        TOKENS_COUNT = _tokens.length;
        tokens = _tokens;
        totalReserves = new uint144[](_tokens.length);
    }

    function _toDynamic(
        uint144[] memory arrayIn
    ) internal pure returns (uint144[] memory arrayOut) {
        arrayOut = new uint144[](arrayIn.length);
        for (uint256 i = 0; i < arrayIn.length; i++) arrayOut[i] = arrayIn[i];
    }

    function _callCalculateRadius(
        uint144[] memory reserves
    ) internal returns (uint144) {
        uint144[] memory dyn = _toDynamic(reserves);
        (bool ok, bytes memory ret) = address(MATH_HELPER).call(
            abi.encodeWithSignature("calculateRadius(uint144[])", dyn)
        );
        if (!ok || ret.length == 0) revert NumericalError();
        return abi.decode(ret, (uint144));
    }

    function _solveQuadraticInvariant(
        uint144 deltaLinear,
        uint144[] memory reserves,
        uint144 tokenInIndex,
        uint144 tokenOutIndex,
        uint144 consolidatedRadius,
        uint144 kCross
    ) internal returns (uint144) {
        uint144[] memory dyn = _toDynamic(reserves);
        (bool ok, bytes memory ret) = address(MATH_HELPER).call(
            abi.encodeWithSignature(
                "solveQuadraticInvariant(uint144,uint144[],uint144,uint144,uint144,uint144)",
                deltaLinear,
                dyn,
                tokenInIndex,
                tokenOutIndex,
                consolidatedRadius,
                kCross
            )
        );
        if (!ok || ret.length == 0) revert NumericalError();
        return abi.decode(ret, (uint144));
    }

    function _callBoundaryTickS(
        uint144 r,
        uint144 k
    ) internal returns (uint144) {
        (bool ok, bytes memory ret) = address(MATH_HELPER).call(
            abi.encodeWithSignature(
                "calculateBoundaryTickS(uint144,uint144)",
                r,
                k
            )
        );
        if (!ok || ret.length == 0) revert NumericalError();
        return abi.decode(ret, (uint144));
    }

    function _sqrtQ96X48(uint144 x) internal returns (uint144) {
        (bool ok, bytes memory ret) = address(MATH_HELPER).call(
            abi.encodeWithSignature("sqrtQ96X48(uint144)", x)
        );
        if (!ok || ret.length == 0) revert NumericalError();
        return abi.decode(ret, (uint144));
    }

    function _calculateTickParams(
        uint144 p,
        uint144 reserveAmount
    ) public returns (uint144, uint144) {
        (bool ok, bytes memory ret) = address(MATH_HELPER).call(
            abi.encodeWithSignature(
                "getTickParameters(uint144,uint144)",
                p,
                reserveAmount
            )
        );
        if (!ok || ret.length == 0) revert NumericalError();
        (uint144 k, uint144 r) = abi.decode(ret, (uint144, uint144));
        emit TickParams(k, r);
        return (k, r);
    }

    function _callSolveTorusInvariant(
        uint144 sumInteriorReserves,
        uint144 interiorR,
        uint144 boundaryR,
        uint144 boundaryK,
        uint144 tokenIn,
        uint144 tokenOut,
        uint144 amountIn,
        uint144[] memory _totalReserves
    ) internal returns (uint144) {
        uint144[] memory dyn = _toDynamic(_totalReserves);
        (bool ok, bytes memory ret) = address(MATH_HELPER).call(
            abi.encodeWithSignature(
                "solveTorusInvariant(uint144,uint144,uint144,uint144,uint144,uint144,uint144,uint144[])",
                sumInteriorReserves,
                interiorR,
                boundaryR,
                boundaryK,
                tokenIn,
                tokenOut,
                amountIn,
                dyn
            )
        );
        if (!ok || ret.length == 0) revert TorusInvariantCallFailed();
        return abi.decode(ret, (uint144));
    }

    /**
     * @notice Gets LP shares for a user in a specific tick
     * @param user User address
     * @return LP shares owned by the user
     */
    function getLpShares(
        uint144 p,
        address user
    ) external view returns (uint144) {
        return activeTicks[p].lpShares[user];
    }

    /**
     * @notice Gets total LP shares for a tick
     * @return Total LP shares for the tick
     */
    function getTotalLpShares(uint144 p) external view returns (uint144) {
        return activeTicks[p].totalLpShares;
    }

    /**
     * @notice Gets the radius of a tick
     * @return Radius of the tick
     */
    function getTickRadius(uint144 p) external view returns (uint144) {
        return activeTicks[p].r;
    }

    /**
     * @notice Gets the liquidity of a tick
     * @return Liquidity of the tick
     */
    function getTickLiquidity(uint144 p) external view returns (uint144) {
        return activeTicks[p].liquidity;
    }

    /**
     * @notice Gets the status of a tick
     * @return Current status of the tick
     */
    function getTickStatus(uint144 p) external view returns (TickStatus) {
        return activeTicks[p].status;
    }

    /**
     * @notice Gets total reserves across all active ticks
     * @return totalReserves Array of total reserves for each token
     */
    function _getTotalReserves() public view returns (uint144[] memory) {
        return totalReserves;
    }
    function setSquareRootN() external  {
        (bool ok, bytes memory ret) = address(MATH_HELPER).call(
            abi.encodeWithSignature("sqrtQ96X48(uint144)", uint144(TOKENS_COUNT))
        );
        if (!ok || ret.length == 0) revert NumericalError();
        uint144 rootn = abi.decode(ret, (uint144));
        ROOT_N = rootn;
    }
    /**
     * @notice Adds liquidity to a tick
     * @param amounts Array of token amounts to add (all must be > 0)
     * @dev Creates new tick if it doesn't exist, updates existing tick otherwise
     */
    function addLiquidity(uint144 p, uint144[] memory amounts) external {
        bool tickExists = activeTicks[p].r > 0;
        uint144 previousTotalLpShares = tickExists
            ? activeTicks[p].totalLpShares
            : 0;
        uint256 previousLiquidity = tickExists ? activeTicks[p].liquidity : 0;

        if (amounts.length != TOKENS_COUNT) {
            revert InvalidLength();
        }
        uint144[] memory reservesForRadiusCalc = new uint144[](TOKENS_COUNT);
        if (!tickExists) {
            reservesForRadiusCalc = amounts;
        } else {
            for (uint256 i = 0; i < TOKENS_COUNT; i++) {
                reservesForRadiusCalc[i] =
                    activeTicks[p].reserves[i] +
                    amounts[i];
            }
        }

        (uint144 k, uint144 newRadius) = _calculateTickParams(
            p,
            reservesForRadiusCalc[0]
        );

        TickStatus tickStatus = checkInvariants(
            k,
            newRadius,
            reservesForRadiusCalc
        );

        if (!tickExists) {
            Tick storage newTick = activeTicks[p];
            newTick.p = p;
            newTick.r = newRadius;
            newTick.k = k;
            newTick.liquidity = uint144(
                (uint256(newRadius) * uint256(newRadius)) >> 48
            );
            for (uint256 i = 0; i < TOKENS_COUNT; i++) {
                newTick.reserves.push(reservesForRadiusCalc[i]);
            }
            newTick.status = tickStatus;
            allP.push(p);
        } else {
            Tick storage tick = activeTicks[p];
            tick.r = newRadius;
            tick.k = k;
            tick.p = p;
            for (uint256 i = 0; i < TOKENS_COUNT; i++) {
                tick.reserves[i] = reservesForRadiusCalc[i];
            }
            tick.liquidity = uint144(
                (uint256(newRadius) * uint256(newRadius)) >> 48
            );
            tick.status = tickStatus;
        }

        for (uint256 i = 0; i < TOKENS_COUNT; i++) {
            if (amounts[i] > 0) {
                tokens[i].safeTransferFrom(
                    msg.sender,
                    address(this),
                    amounts[i] >> 48
                );
                totalReserves[i] += amounts[i];
            } else {
                revert InvalidAmounts();
            }
        }

        uint144 lpShares;
        if (!tickExists || previousTotalLpShares == 0) {
            lpShares = uint144((uint256(newRadius) * uint256(newRadius)) >> 48);
        } else {
            uint144 newLiquidity = activeTicks[p].liquidity;
            lpShares = uint144(
                ((uint256(newLiquidity) - uint256(previousLiquidity)) *
                    uint256(previousTotalLpShares)) / previousLiquidity
            );
        }

        activeTicks[p].lpShares[msg.sender] += lpShares;
        activeTicks[p].totalLpShares += lpShares;

        uint144[] memory shiftedAmounts = new uint144[](amounts.length);
        for (uint256 i = 0; i < amounts.length; i++) {
            shiftedAmounts[i] = amounts[i] >> 48;
        }
        emit LiquidityAdded(shiftedAmounts, lpShares, msg.sender, p >> 48);
    }

    /**
     * @notice Removes liquidity from a tick
     * @param lpSharesToRemove Number of LP shares to remove
     * @dev Proportionally removes reserves and updates tick state
     */
    function removeLiquidity(uint144 p, uint144 lpSharesToRemove) external {
        if (lpSharesToRemove == 0) revert InvalidAmounts();

        Tick storage tick = activeTicks[p];
        if (tick.r == 0) revert InvalidKValue();

        uint144 userLpShares = tick.lpShares[msg.sender];
        if (userLpShares < lpSharesToRemove) revert InsufficientLiquidity();

        uint144 removalProportion = (lpSharesToRemove << 48) /
            tick.totalLpShares;

        uint144[] memory amountsToReturn = new uint144[](TOKENS_COUNT);
        uint144[] memory newReserves = new uint144[](TOKENS_COUNT);
        for (uint256 i = 0; i < TOKENS_COUNT; i++) {
            amountsToReturn[i] = ((tick.reserves[i] * removalProportion) >> 48);
            newReserves[i] = tick.reserves[i] - amountsToReturn[i];
        }

        (uint144 newRadius, uint144 k) = _calculateTickParams(
            p,
            newReserves[0]
        );

        tick.r = newRadius;
        tick.k = k;
        tick.reserves = newReserves;
        tick.liquidity = uint144(
            (uint256(newRadius) * uint256(newRadius)) >> 48
        );
        tick.lpShares[msg.sender] -= lpSharesToRemove;
        tick.totalLpShares -= lpSharesToRemove;

        if (tick.totalLpShares == 0) {
            _removeTickFromAllP(p);
        }

        TickStatus tickStatus = checkInvariants(newRadius, k, newReserves);
        tick.status = tickStatus;

        for (uint256 i = 0; i < TOKENS_COUNT; i++) {
            if (amountsToReturn[i] > 0) {
                tokens[i].safeTransfer(msg.sender, amountsToReturn[i] >> 48);
                totalReserves[i] -= amountsToReturn[i];
            }
        }

        uint144[] memory shiftedAmounts = new uint144[](TOKENS_COUNT);
        for (uint256 i = 0; i < TOKENS_COUNT; i++) {
            shiftedAmounts[i] = amountsToReturn[i] >> 48;
        }
        emit LiquidityRemoved(shiftedAmounts, msg.sender, p >> 48);
    }

    function checkInvariants(
        uint144 k,
        uint144 r,
        uint144[] memory amounts
    ) internal view returns (TickStatus) {
        uint256 sumOfDifferenceOfReserves = 0;
        uint256 sum = 0; 
        for (uint256 i = 0; i < TOKENS_COUNT; i++) {
            sum += uint256(amounts[i]);
            uint256 difference = uint256(r) - uint256(amounts[i]);
            sumOfDifferenceOfReserves += (difference * difference) >> 48;
        }
        uint256 invariant = (uint256(r) * uint256(r)) >> 48;
        uint256 tolerance = invariant / 1000; // 0.1% tolerance
        if (
            sumOfDifferenceOfReserves > invariant + tolerance ||
            sumOfDifferenceOfReserves < invariant - tolerance
        ) {
            revert UnsatisfiedInvariant();
        }
        uint256 lhs = (sum << 48) / ROOT_N;
        return
            (lhs < uint256(k) + tolerance && lhs > uint256(k) - tolerance)
                ? TickStatus.Boundary
                : TickStatus.Interior;
    }

    /**
     * @notice Removes a tick from the allP array
     * @param p The tick parameter to remove
     * @dev Helper function to maintain the allP array when ticks are emptied
     */
    function _removeTickFromAllP(uint144 p) internal {
        for (uint256 i = 0; i < allP.length; i++) {
            if (allP[i] == p) {
                allP[i] = allP[allP.length - 1];
                allP.pop();
                break;
            }
        }
    }

      function swap(
        uint144 amountIn,
        uint144 tokenIn,
        uint144 tokenOut,
        uint144 minAmountOut
    ) external {
        if (tokenIn == tokenOut) {
            revert SameToken();
        }

        tokens[tokenIn].safeTransferFrom(
            msg.sender,
            address(this),
            amountIn >> 48
        );

        uint144 amountRemaining = amountIn;

        uint144 totalAmountOut = 0;

        while (amountRemaining >= (1 << 48)) {
            (
                ConsolidatedTickData memory interiorTickData,

            ) = _getConsolidatedTickData();

            uint144 alphaIntNormBeforeSwap = computeAlphaIntNorm(
                interiorTickData
            );

            uint144 estimatedAmountOut = _calculateSwapOutput(
                interiorTickData.totalReserves,
                tokenIn,
                tokenOut,
                amountRemaining
            );

            uint144[] memory hypotheticalReserves = new uint144[](TOKENS_COUNT);
            for (uint256 i = 0; i < TOKENS_COUNT; i++) {
                hypotheticalReserves[i] = interiorTickData.totalReserves[i];
            }
            hypotheticalReserves[tokenIn] += amountRemaining;
            if (estimatedAmountOut > hypotheticalReserves[tokenOut]) {
                revert InsufficientLiquidity();
            }
            hypotheticalReserves[tokenOut] -= estimatedAmountOut;

            uint144 alphaIntNormAfterSwap = 0;
            for (uint256 i = 0; i < TOKENS_COUNT; i++) {
                alphaIntNormAfterSwap += uint144(
                    (uint256(hypotheticalReserves[i]) << 48) /
                        interiorTickData.consolidatedRadius
                );
            }

            (uint144 kIntMin, uint144 kBoundMax) = calculateKBounds();

            bool crossing = (alphaIntNormAfterSwap > kBoundMax ||
                alphaIntNormAfterSwap < kIntMin);

            if (!crossing) {
                uint144 amountOutFull = estimatedAmountOut;

                _updateReserves(
                    tokenIn,
                    amountRemaining,
                    tokenOut,
                    amountOutFull
                );

                tokens[tokenOut].safeTransfer(msg.sender, amountOutFull);

                totalAmountOut += amountOutFull;
                amountRemaining = 0;
                break;
            }

            uint144 kCross = determineDirectionAndKCross(
                alphaIntNormBeforeSwap,
                alphaIntNormAfterSwap,
                kIntMin,
                kBoundMax
            );

            uint144 delta = computeDeltaToCrossBoundary(
                alphaIntNormBeforeSwap,
                kCross,
                amountRemaining,
                interiorTickData.consolidatedRadius,
                interiorTickData.totalReserves,
                tokenIn,
                tokenOut
            );

            uint144 amountOutPartial = _calculateSwapOutput(
                interiorTickData.totalReserves,
                tokenIn,
                tokenOut,
                delta
            );

            _updateReserves(tokenIn, delta, tokenOut, amountOutPartial);

            tokens[tokenOut].safeTransfer(msg.sender, amountOutPartial);

            totalAmountOut += amountOutPartial;
            amountRemaining -= delta;

            _flipTickStatus(kCross);
        }

        if (totalAmountOut < minAmountOut) revert SlippageExceeded();

        emit Swap(
            msg.sender,
            tokenIn,
            tokenOut,
            amountIn,
            totalAmountOut,
            amountIn - amountRemaining
        );
    }


    function _updateReserves(
        uint144 tokenIn,
        uint144 deltaIn,
        uint144 tokenOut,
        uint144 deltaOut
    ) internal {
        totalReserves[tokenIn] += deltaIn;
        if (totalReserves[tokenOut] < deltaOut) {
            revert InvalidReserves();
        }
        totalReserves[tokenOut] -= deltaOut;

        uint256 interiorLiquiditySum = 0;
        for (uint256 i = 0; i < allP.length; i++) {
            uint144 p = allP[i];
            Tick storage tick = activeTicks[p];
            if (tick.status == TickStatus.Interior) {
                interiorLiquiditySum += tick.liquidity;
            }
        }
        if (interiorLiquiditySum == 0) {
            revert NoInteriorLiquidity();
        }

        for (uint256 i = 0; i < allP.length; i++) {
            uint144 p = allP[i];
            Tick storage tick = activeTicks[p];
            if (tick.status == TickStatus.Interior) {
                uint256 share = (uint256(tick.liquidity) << 48) /
                    interiorLiquiditySum;

                uint144 deltaInTick = uint144((uint256(deltaIn) * share) >> 48);
                uint144 deltaOutTick = uint144(
                    (uint256(deltaOut) * share) >> 48
                );

                tick.reserves[tokenIn] += deltaInTick;
                if (tick.reserves[tokenOut] < deltaOut) {
                    revert InvalidReserves();
                }
                tick.reserves[tokenOut] -= deltaOutTick;
            }
        }
    }

    function _flipTickStatus(uint144 kCross) internal {
        for (uint256 i = 0; i < allP.length; i++) {
            uint144 p = allP[i];
            Tick storage tick = activeTicks[p];
            if (tick.r == 0) continue;
            uint144 kNorm = uint144((uint256(tick.k) << 48) / tick.r);

            if (kNorm == kCross) {
                if (tick.status == TickStatus.Interior) {
                    tick.status = TickStatus.Boundary;
                } else if (tick.status == TickStatus.Boundary) {
                    tick.status = TickStatus.Interior;
                }
                break;
            }
        }
    }

    function determineDirectionAndKCross(
        uint144 alphaIntNormBefore,
        uint144 alphaIntNormAfter,
        uint144 kIntMin,
        uint144 kBoundMax
    ) internal pure returns (uint144 kCross) {
        if (alphaIntNormAfter > alphaIntNormBefore) {
            kCross = kIntMin;
        } else {
            kCross = kBoundMax;
        }
    }

    function computeDeltaToCrossBoundary(
        uint144 alphaCurrent,
        uint144 kCross,
        uint144 amountRemaining,
        uint144 consolidatedRadius,
        uint144[] memory reserves,
        uint144 tokenIn,
        uint144 tokenOut
    ) internal returns (uint144 deltaCross) {
        uint144 deltaLinear = uint144(
            (uint256(consolidatedRadius) * absDiff(alphaCurrent, kCross)) >> 48
        );

        deltaCross = _solveQuadraticInvariant(
            deltaLinear,
            reserves,
            tokenIn,
            tokenOut,
            consolidatedRadius,
            kCross
        );

        if (deltaCross > amountRemaining) {
            deltaCross = amountRemaining;
        }
    }

    function absDiff(uint144 a, uint144 b) internal pure returns (uint144) {
        if (a > b) {
            return a - b;
        } else {
            return b - a;
        }
    }

    function calculateKBounds()
        internal
        view
        returns (uint144 kIntMin, uint144 kBoundMax)
    {
        kIntMin = type(uint144).max;
        kBoundMax = 0;

        for (uint256 i = 0; i < allP.length; i++) {
            uint144 p = allP[i];
            Tick storage tick = activeTicks[p];

            if (tick.r == 0) continue;

            uint144 kNorm = uint144((uint256(tick.k) << 48) / tick.r);

            if (tick.status == TickStatus.Interior) {
                if (kNorm < kIntMin) {
                    kIntMin = kNorm;
                }
            } else if (tick.status == TickStatus.Boundary) {
                if (kNorm > kBoundMax) {
                    kBoundMax = kNorm;
                }
            }
        }

        if (kIntMin == type(uint144).max) {
            kIntMin = 0;
        }
    }

    function _calculateSwapOutput(
        uint144[] memory reserves,
        uint144 tokenIn,
        uint144 tokenOut,
        uint144 amountIn
    ) internal returns (uint144) {
        (
            ConsolidatedTickData memory interiorData,
            ConsolidatedTickData memory boundaryData
        ) = _getConsolidatedTickData();

        uint144 sumInteriorReserves = 0;
        for (uint256 i = 0; i < TOKENS_COUNT; i++) {
            sumInteriorReserves += interiorData.totalReserves[i];
        }

        uint144 amount = _callSolveTorusInvariant(
            sumInteriorReserves,
            interiorData.consolidatedRadius,
            boundaryData.consolidatedRadius,
            boundaryData.totalKBound,
            tokenIn,
            tokenOut,
            amountIn,
            totalReserves
        );

        if (amount >= reserves[tokenOut]) revert InsufficientLiquidity();
        return amount;
    }

    function _getConsolidatedTickData()
        internal
        returns (
            ConsolidatedTickData memory interiorData,
            ConsolidatedTickData memory boundaryData
        )
    {
        interiorData.totalReserves = new uint144[](TOKENS_COUNT);
        interiorData.sumSquaredReserves = new uint144[](TOKENS_COUNT);

        boundaryData.totalReserves = new uint144[](TOKENS_COUNT);
        boundaryData.sumSquaredReserves = new uint144[](TOKENS_COUNT);
        for (uint256 i = 0; i < allP.length; i++) {
            uint144 p = allP[i];
            Tick storage tick = activeTicks[p];

            if (tick.r == 0) continue;

            if (tick.status == TickStatus.Interior) {
                interiorData.consolidatedRadius += tick.r;
                interiorData.totalLiquidity += tick.liquidity;
                interiorData.tickCount++;
                for (uint256 j = 0; j < TOKENS_COUNT; j++) {
                    interiorData.totalReserves[j] += tick.reserves[j];
                }
            } else {
                uint144 s = _callBoundaryTickS(tick.r, tick.k);
                boundaryData.consolidatedRadius += s;
                boundaryData.totalLiquidity += tick.liquidity;
                boundaryData.tickCount++;
                boundaryData.totalKBound += tick.k;
                for (uint256 j = 0; j < TOKENS_COUNT; j++) {
                    boundaryData.totalReserves[j] += tick.reserves[j];
                }
            }
        }
    }

    function computeAlphaIntNorm(
        ConsolidatedTickData memory tickData
    ) internal view returns (uint144 alpha) {
        alpha = 0;
        for (uint256 i = 0; i < TOKENS_COUNT; i++) {
            alpha +=
                (tickData.totalReserves[i] << 48) /
                ((tickData.consolidatedRadius * ROOT_N) >> 48);
        }
    }
}