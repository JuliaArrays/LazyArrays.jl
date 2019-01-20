using LazyArrays, LinearAlgebra, FillArrays, Test
    import LazyArrays: MemoryLayout, DenseRowMajor, DenseColumnMajor, StridedLayout,
                            ConjLayout, RowMajor, ColumnMajor, UnknownLayout,
                            SymmetricLayout, HermitianLayout, UpperTriangularLayout,
                            UnitUpperTriangularLayout, LowerTriangularLayout,
                            UnitLowerTriangularLayout, ScalarLayout,
                            hermitiandata, symmetricdata, FillLayout, ZerosLayout,
                            VcatLayout, BroadcastLayout, Add, AddArray, ApplyLayout

struct FooBar end
struct FooNumber <: Number end

@testset "MemoryLayout" begin
    @testset "Trivial" begin
        @test MemoryLayout(1.0) == MemoryLayout(1) == MemoryLayout(FooNumber()) == ScalarLayout()
        @test MemoryLayout(FooBar()) == UnknownLayout()

        A = randn(6)
        @test MemoryLayout(A) == MemoryLayout(Base.ReshapedArray(A,(2,3),())) == MemoryLayout(reinterpret(Float32,A)) ==
            DenseColumnMajor()

    end

    @testset "adjoint and transpose MemoryLayout" begin
        A = [1.0 2; 3 4]
        @test MemoryLayout(A') == DenseRowMajor()
        @test MemoryLayout(transpose(A)) == DenseRowMajor()
        B = [1.0+im 2; 3 4]
        @test MemoryLayout(B') == ConjLayout(DenseRowMajor())
        @test MemoryLayout(transpose(B)) == DenseRowMajor()
        VA = view(A, 1:1, 1:1)
        @test MemoryLayout(VA') == RowMajor()
        @test MemoryLayout(transpose(VA)) == RowMajor()
        VB = view(B, 1:1, 1:1)
        @test MemoryLayout(VB') == ConjLayout(RowMajor())
        @test MemoryLayout(transpose(VB)) == RowMajor()
        VA = view(A, 1:2:2, 1:2:2)
        @test MemoryLayout(VA') == StridedLayout()
        @test MemoryLayout(transpose(VA)) == StridedLayout()
        VB = view(B, 1:2:2, 1:2:2)
        @test MemoryLayout(VB') == ConjLayout(StridedLayout())
        @test MemoryLayout(transpose(VB)) == StridedLayout()
        VA2 = view(A, [1,2], :)
        @test MemoryLayout(VA2') == UnknownLayout()
        @test MemoryLayout(transpose(VA2)) == UnknownLayout()
        VB2 = view(B, [1,2], :)
        @test MemoryLayout(VB2') == UnknownLayout()
        @test MemoryLayout(transpose(VB2)) == UnknownLayout()
        VAc = view(A', 1:1, 1:1)
        @test MemoryLayout(VAc) == RowMajor()
        VAt = view(transpose(A), 1:1, 1:1)
        @test MemoryLayout(VAt) == RowMajor()
        VBc = view(B', 1:1, 1:1)
        @test MemoryLayout(VBc) == ConjLayout(RowMajor())
        VBt = view(transpose(B), 1:1, 1:1)
        @test MemoryLayout(VBt) == RowMajor()
    end


    @testset "Symmetric/Hermitian MemoryLayout" begin
         A = [1.0 2; 3 4]
         @test MemoryLayout(Symmetric(A)) == SymmetricLayout(DenseColumnMajor(),'U')
         @test MemoryLayout(Hermitian(A)) == SymmetricLayout(DenseColumnMajor(),'U')
         @test MemoryLayout(Transpose(Symmetric(A))) == SymmetricLayout(DenseColumnMajor(),'U')
         @test MemoryLayout(Transpose(Hermitian(A))) == SymmetricLayout(DenseColumnMajor(),'U')
         @test MemoryLayout(Adjoint(Symmetric(A))) == SymmetricLayout(DenseColumnMajor(),'U')
         @test MemoryLayout(Adjoint(Hermitian(A))) == SymmetricLayout(DenseColumnMajor(),'U')
         @test MemoryLayout(view(Symmetric(A),:,:)) == SymmetricLayout(DenseColumnMajor(),'U')
         @test MemoryLayout(view(Hermitian(A),:,:)) == SymmetricLayout(DenseColumnMajor(),'U')
         @test MemoryLayout(Symmetric(A')) == SymmetricLayout(DenseRowMajor(),'U')
         @test MemoryLayout(Hermitian(A')) == SymmetricLayout(DenseRowMajor(),'U')
         @test MemoryLayout(Symmetric(transpose(A))) == SymmetricLayout(DenseRowMajor(),'U')
         @test MemoryLayout(Hermitian(transpose(A))) == SymmetricLayout(DenseRowMajor(),'U')

         @test symmetricdata(Symmetric(A)) ≡ A
         @test symmetricdata(Hermitian(A)) ≡ A
         @test symmetricdata(Transpose(Symmetric(A))) ≡ A
         @test symmetricdata(Transpose(Hermitian(A))) ≡ A
         @test symmetricdata(Adjoint(Symmetric(A))) ≡ A
         @test symmetricdata(Adjoint(Hermitian(A))) ≡ A
         @test symmetricdata(view(Symmetric(A),:,:)) ≡ A
         @test symmetricdata(view(Hermitian(A),:,:)) ≡ A
         @test symmetricdata(Symmetric(A')) ≡ A'
         @test symmetricdata(Hermitian(A')) ≡ A'
         @test symmetricdata(Symmetric(transpose(A))) ≡ transpose(A)
         @test symmetricdata(Hermitian(transpose(A))) ≡ transpose(A)

         B = [1.0+im 2; 3 4]
         @test MemoryLayout(Symmetric(B)) == SymmetricLayout(DenseColumnMajor(),'U')
         @test MemoryLayout(Hermitian(B)) == HermitianLayout(DenseColumnMajor(),'U')
         @test MemoryLayout(Transpose(Symmetric(B))) == SymmetricLayout(DenseColumnMajor(),'U')
         @test MemoryLayout(Transpose(Hermitian(B))) == UnknownLayout()
         @test MemoryLayout(Adjoint(Symmetric(B))) == UnknownLayout()
         @test MemoryLayout(Adjoint(Hermitian(B))) == HermitianLayout(DenseColumnMajor(),'U')
         @test MemoryLayout(view(Symmetric(B),:,:)) == SymmetricLayout(DenseColumnMajor(),'U')
         @test MemoryLayout(view(Hermitian(B),:,:)) == HermitianLayout(DenseColumnMajor(),'U')
         @test MemoryLayout(Symmetric(B')) == UnknownLayout()
         @test MemoryLayout(Hermitian(B')) == UnknownLayout()
         @test MemoryLayout(Symmetric(transpose(B))) == SymmetricLayout(DenseRowMajor(),'U')
         @test MemoryLayout(Hermitian(transpose(B))) == HermitianLayout(DenseRowMajor(),'U')

         @test symmetricdata(Symmetric(B)) ≡ B
         @test hermitiandata(Hermitian(B)) ≡ B
         @test symmetricdata(Transpose(Symmetric(B))) ≡ B
         @test hermitiandata(Adjoint(Hermitian(B))) ≡ B
         @test symmetricdata(view(Symmetric(B),:,:)) ≡ B
         @test hermitiandata(view(Hermitian(B),:,:)) ≡ B
         @test symmetricdata(Symmetric(B')) ≡ B'
         @test hermitiandata(Hermitian(B')) ≡ B'
         @test symmetricdata(Symmetric(transpose(B))) ≡ transpose(B)
         @test hermitiandata(Hermitian(transpose(B))) ≡ transpose(B)
     end


    @testset "triangular MemoryLayout" begin
        A = [1.0 2; 3 4]
        B = [1.0+im 2; 3 4]
        for (TriType, TriLayout, TriLayoutTrans) in ((UpperTriangular, UpperTriangularLayout, LowerTriangularLayout),
                              (UnitUpperTriangular, UnitUpperTriangularLayout, UnitLowerTriangularLayout),
                              (LowerTriangular, LowerTriangularLayout, UpperTriangularLayout),
                              (UnitLowerTriangular, UnitLowerTriangularLayout, UnitUpperTriangularLayout))
            @test MemoryLayout(TriType(A)) == TriLayout(DenseColumnMajor())
            @test MemoryLayout(TriType(transpose(A))) == TriLayout(DenseRowMajor())
            @test MemoryLayout(TriType(A')) == TriLayout(DenseRowMajor())
            @test MemoryLayout(transpose(TriType(A))) == TriLayoutTrans(DenseRowMajor())
            @test MemoryLayout(TriType(A)') == TriLayoutTrans(DenseRowMajor())

            @test MemoryLayout(TriType(B)) == TriLayout(DenseColumnMajor())
            @test MemoryLayout(TriType(transpose(B))) == TriLayout(DenseRowMajor())
            @test MemoryLayout(TriType(B')) == TriLayout(ConjLayout(DenseRowMajor()))
            @test MemoryLayout(transpose(TriType(B))) == TriLayoutTrans(DenseRowMajor())
            @test MemoryLayout(TriType(B)') == TriLayoutTrans(ConjLayout(DenseRowMajor()))
        end

        @test MemoryLayout(UpperTriangular(B)') == MemoryLayout(LowerTriangular(B'))
    end

    @testset "Fill and Vcat" begin
        @test MemoryLayout(Fill(1,10)) == FillLayout()
        @test MemoryLayout(Ones(10)) == FillLayout()
        @test MemoryLayout(Zeros(10)) == ZerosLayout()
        @test MemoryLayout(Vcat(Ones(10),Zeros(10))) == VcatLayout((FillLayout(), ZerosLayout()))
        @test MemoryLayout(Vcat([1.],Zeros(10))) == VcatLayout((DenseColumnMajor(), ZerosLayout()))
    end

    @testset "BroadcastArray" begin
        A = [1.0 2; 3 4]
        @test MemoryLayout(BroadcastArray(+, A, Fill(0, (2, 2)), Zeros(2, 2))) ==
            BroadcastLayout(+, (DenseColumnMajor(), FillLayout(), ZerosLayout()))
    end

    @testset "ApplyArray" begin
        A = [1.0 2; 3 4]
        @test eltype(AddArray(A, Fill(0, (2, 2)), Zeros(2, 2))) == Float64
        @test MemoryLayout(AddArray(A, Fill(0, (2, 2)), Zeros(2, 2))) ==
            ApplyLayout(+, (DenseColumnMajor(), FillLayout(), ZerosLayout()))
    end
end
