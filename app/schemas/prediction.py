from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    mdvp_fo: float = Field(alias="MDVP:Fo(Hz)")
    mdvp_fhi: float = Field(alias="MDVP:Fhi(Hz)")
    mdvp_flo: float = Field(alias="MDVP:Flo(Hz)")
    mdvp_jitter_percent: float = Field(alias="MDVP:Jitter(%)")
    mdvp_jitter_abs: float = Field(alias="MDVP:Jitter(Abs)")
    mdvp_rap: float = Field(alias="MDVP:RAP")
    mdvp_ppq: float = Field(alias="MDVP:PPQ")
    jitter_ddp: float = Field(alias="Jitter:DDP")
    mdvp_shimmer: float = Field(alias="MDVP:Shimmer")
    mdvp_shimmer_db: float = Field(alias="MDVP:Shimmer(dB)")
    shimmer_apq3: float = Field(alias="Shimmer:APQ3")
    shimmer_apq5: float = Field(alias="Shimmer:APQ5")
    mdvp_apq: float = Field(alias="MDVP:APQ")
    shimmer_dda: float = Field(alias="Shimmer:DDA")
    nhr: float = Field(alias="NHR")
    hnr: float = Field(alias="HNR")
    rpde: float = Field(alias="RPDE")
    dfa: float = Field(alias="DFA")
    spread1: float = Field(alias="spread1")
    spread2: float = Field(alias="spread2")
    d2: float = Field(alias="D2")
    ppe: float = Field(alias="PPE")

    model_config = {
        "populate_by_name": True
    }

    def to_model_features(self) -> dict:
        """Convert API fields back to model feature names."""

        return {
            "MDVP:Fo(Hz)": self.mdvp_fo,
            "MDVP:Fhi(Hz)": self.mdvp_fhi,
            "MDVP:Flo(Hz)": self.mdvp_flo,
            "MDVP:Jitter(%)": self.mdvp_jitter_percent,
            "MDVP:Jitter(Abs)": self.mdvp_jitter_abs,
            "MDVP:RAP": self.mdvp_rap,
            "MDVP:PPQ": self.mdvp_ppq,
            "Jitter:DDP": self.jitter_ddp,
            "MDVP:Shimmer": self.mdvp_shimmer,
            "MDVP:Shimmer(dB)": self.mdvp_shimmer_db,
            "Shimmer:APQ3": self.shimmer_apq3,
            "Shimmer:APQ5": self.shimmer_apq5,
            "MDVP:APQ": self.mdvp_apq,
            "Shimmer:DDA": self.shimmer_dda,
            "NHR": self.nhr,
            "HNR": self.hnr,
            "RPDE": self.rpde,
            "DFA": self.dfa,
            "spread1": self.spread1,
            "spread2": self.spread2,
            "D2": self.d2,
            "PPE": self.ppe,
        }
