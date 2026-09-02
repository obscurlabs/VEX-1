// SPDX-License-Identifier: MIT
pragma solidity 0.8.26;

/// @title IdentityAnchor
/// @notice Anchors the SHA-256 fingerprint of an off-chain evidence manifest.
///
/// @dev What this contract does NOT do, deliberately:
///        - no face recognition, no web search, no evidence processing
///        - stores no images, no face embeddings, no raw search results
///        - stores no personal or biometric data of any kind
///
///      It records two 32-byte values plus the provenance the chain supplies
///      for free (sender, block timestamp, block number). That is enough to
///      prove a given fingerprint existed at a given time and was submitted by
///      a given address.
///
///      It proves NOTHING about whether the underlying evidence is truthful,
///      and NOTHING about anyone's real-world identity.
///
///      Duplicate policy: an investigation may be anchored exactly once, and
///      an evidence hash may be anchored exactly once. Both are rejected on a
///      second attempt so a fingerprint can never map to two conflicting
///      records. Re-running an investigation therefore requires a new
///      investigation id, which is what a genuinely new observation is.
contract IdentityAnchor {
    struct EvidenceRecord {
        bytes32 evidenceHash; // SHA-256 of the canonical evidence manifest
        address submitter;    // who anchored it
        uint64 timestamp;     // block timestamp at anchoring
        uint64 blockNumber;   // block height at anchoring
    }

    /// @dev keccak256(investigation id) => record
    mapping(bytes32 => EvidenceRecord) private _records;
    /// @dev evidence hash => investigation id, so a hash resolves back to its record
    mapping(bytes32 => bytes32) private _hashToInvestigation;

    uint256 public totalAnchored;

    event EvidenceAnchored(
        bytes32 indexed investigationId,
        bytes32 indexed evidenceHash,
        address indexed submitter,
        uint64 timestamp,
        uint64 blockNumber
    );

    error ZeroValue();
    error InvestigationAlreadyAnchored(bytes32 investigationId);
    error EvidenceHashAlreadyAnchored(bytes32 evidenceHash, bytes32 investigationId);
    error NotAnchored(bytes32 investigationId);

    /// @notice Anchor an evidence fingerprint.
    /// @param investigationId keccak256 of the investigation id string
    /// @param evidenceHash    SHA-256 of the canonical evidence manifest
    function anchorEvidence(bytes32 investigationId, bytes32 evidenceHash) external {
        if (investigationId == bytes32(0) || evidenceHash == bytes32(0)) {
            revert ZeroValue();
        }
        if (_records[investigationId].evidenceHash != bytes32(0)) {
            revert InvestigationAlreadyAnchored(investigationId);
        }
        bytes32 existing = _hashToInvestigation[evidenceHash];
        if (existing != bytes32(0)) {
            revert EvidenceHashAlreadyAnchored(evidenceHash, existing);
        }

        _records[investigationId] = EvidenceRecord({
            evidenceHash: evidenceHash,
            submitter: msg.sender,
            timestamp: uint64(block.timestamp),
            blockNumber: uint64(block.number)
        });
        _hashToInvestigation[evidenceHash] = investigationId;
        unchecked {
            ++totalAnchored;
        }

        emit EvidenceAnchored(
            investigationId,
            evidenceHash,
            msg.sender,
            uint64(block.timestamp),
            uint64(block.number)
        );
    }

    /// @notice Read an anchored record. Reverts if the investigation is unknown.
    function getEvidence(bytes32 investigationId)
        external
        view
        returns (bytes32 evidenceHash, address submitter, uint64 timestamp, uint64 blockNumber)
    {
        EvidenceRecord storage record = _records[investigationId];
        if (record.evidenceHash == bytes32(0)) {
            revert NotAnchored(investigationId);
        }
        return (record.evidenceHash, record.submitter, record.timestamp, record.blockNumber);
    }

    /// @notice Compare a supplied fingerprint against the anchored one.
    /// @return true only when the investigation is anchored AND matches.
    function verifyEvidence(bytes32 investigationId, bytes32 expectedHash)
        external
        view
        returns (bool)
    {
        if (expectedHash == bytes32(0)) {
            return false;
        }
        return _records[investigationId].evidenceHash == expectedHash;
    }

    /// @notice Whether an investigation has been anchored.
    function isAnchored(bytes32 investigationId) external view returns (bool) {
        return _records[investigationId].evidenceHash != bytes32(0);
    }

    /// @notice Resolve an evidence hash back to its investigation id.
    /// @return bytes32(0) when that hash has never been anchored.
    function investigationForHash(bytes32 evidenceHash) external view returns (bytes32) {
        return _hashToInvestigation[evidenceHash];
    }
}
