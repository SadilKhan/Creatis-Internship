# Creatis-Internship
For Master Thesis in Creatis Lab, Insa Lyon

```
class MarchingCubesAlgorithm:
    def __init__(self,model,depth=4):
        self.model = model
        self.depth = 4
        
    def load(self,imagePath):
        image=nib.load(imagePath)
        self.imageData=image.get_fdata()
        self.invrotMat = torch.inverse(self.edge.affine[:3, :3])
        self.transMat = self.edge.affine[:3, 3]

    def findShape(self,imagePath,oldPoints,old_nbr_intensity,label,mean,norm,initMinPoint,initMaxPoint):
        minPoint = initMinPoint
        maxPoint = initMaxPoint
        points=self.calculateGrid(minPoint,maxPoint)
        steps=points[1]-points[0]
        orignalPoints=self.findOrignalPoints(points,mean,norm,self.invrotMat,self.transMat)
        nbr_intensity=[find_neighbor_points(orignalPoints[i,0],orignalPoints[i,1],orignalPoints[i,2],self.imageData)]
        nbr_intensity=torch.vstack(nbr_intensity)
        labelPoints=[]
        new_nbr_intensity=torch.concat([old_nbr_intensity,nbr_intensity])
        newPoints=torch.concat([oldPoints,points.unsqueeze(0)],dim=1)
        for i in range(self.depth):
            steps/=2
            scores,_,_,_,_=self.model(newPoints,new_nbr_intensity)
            labels=scores.argmax(dim=1)
            labelPos=(labels==label).nonzero()[:,1]


    def calculateGrid(self,minPoint,maxPoint,gridPoints=10):
        allPointsX=torch.linspace(minPoint[0],maxPoint[0],gridPoints).reshape(-1,1)
        allPointsY=torch.linspace(minPoint[1],maxPoint[1],gridPoints).reshape(-1,1)
        allPointsZ=torch.linspace(minPoint[2],maxPoint[2],gridPoints).reshape(-1,1)
        allPoints=torch.concat([allPointsX,allPointsY,allPointsZ],axis=1)
        return allPoints
    
    def findOrignalPoints(self,points,mean,norm,invrotMat,transMat):
        orignalPoints=((points*norm+mean)-transMat)@invrotMat
        orignalPoints=torch.ceil(orignalPoints)
        orignalPoints=orignalPoints.long()
        return orignalPoints
        ```
